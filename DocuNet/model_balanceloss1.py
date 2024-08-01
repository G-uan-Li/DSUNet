import torch
import torch.nn as nn
from opt_einsum import contract
from long_seq import process_long_input
from losses import balanced_loss as ATLoss
import torch.nn.functional as F
from allennlp.modules.matrix_attention import (
    DotProductMatrixAttention,
    CosineMatrixAttention,
    BilinearMatrixAttention,
)

# from element_wise import ElementWiseMatrixAttention
from attn_unet import AttentionUNet
from vision_transformer import SwinUnet


class DocREModel(nn.Module):
    """
    DocNet
    """

    def __init__(self, config, args, model, emb_size=768, block_size=64, num_labels=-1):
        super().__init__()
        self.config = config  # 预训练模型的 config
        self.bert_model = model  # 预训练模型
        self.hidden_size = config.hidden_size
        self.entity_pair_block_size = (
            args.entity_pair_block_size
        )  # block size for entity pair representation, different from the block size in classification module
        # self.grouped_projection = nn.Parameter(
        #     torch.randn(
        #         config.hidden_size // self.entity_pair_block_size,
        #         self.entity_pair_block_size,
        #         self.entity_pair_block_size,
        #     )
        # )

        self.grouped_projection = nn.Linear(
            config.hidden_size * self.entity_pair_block_size,
            config.hidden_size,
            bias=False,
        )

        self.loss_fnt = ATLoss()

        self.head_extractor = nn.Linear(
            1 * config.hidden_size + args.unet_out_dim, emb_size
        )
        self.tail_extractor = nn.Linear(
            1 * config.hidden_size + args.unet_out_dim, emb_size
        )

        self.enhanced_head_extractor = nn.Linear(2 * config.hidden_size, emb_size)
        self.enhanced_tail_extractor = nn.Linear(2 * config.hidden_size, emb_size)
        # self.head_extractor = nn.Linear(1 * config.hidden_size , emb_size)
        # self.tail_extractor = nn.Linear(1 * config.hidden_size , emb_size)
        self.bilinear = nn.Linear(emb_size * block_size, config.num_labels)

        self.emb_size = emb_size
        self.block_size = block_size
        self.num_labels = num_labels

        self.bertdrop = nn.Dropout(0.6)
        self.unet_in_dim = args.unet_in_dim
        self.unet_out_dim = args.unet_in_dim  # 3
        # print(
        #     "self.unet_in_dim: {}, self.unet_out_dim: {}".format(
        #         self.unet_in_dim, self.unet_out_dim
        #     )
        # )  # 3, 3
        self.liner = nn.Linear(config.hidden_size, args.unet_in_dim)
        self.min_height = args.max_height  # sh 脚本里面指定了每个数据集的最大实体数(max_height)
        self.channel_type = args.channel_type
        if not args.is_using_swin:
            self.segmentation_net = AttentionUNet(
                input_channels=args.unet_in_dim,
                class_number=args.unet_out_dim,
                down_channel=args.down_dim,
            )
        else:
            self.segmentation_net = SwinUnet(args)
        # print(self.segmentation_net)

    def encode(self, input_ids, attention_mask, entity_pos):
        config = self.config
        if config.transformer_type == "bert":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id]
        elif config.transformer_type == "roberta":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id, config.sep_token_id]
        sequence_output, attention = process_long_input(
            self.bert_model, input_ids, attention_mask, start_tokens, end_tokens
        )
        return sequence_output, attention

    def get_hrt(self, sequence_output, attention, entity_pos, hts):
        offset = 1 if self.config.transformer_type in ["bert", "roberta"] else 0
        bs, h, _, c = attention.size()
        # ne = max([len(x) for x in entity_pos])  # 本次bs中的最大实体数

        hss, tss, rss = [], [], []
        entity_es = []
        entity_as = []
        for i in range(len(entity_pos)):
            entity_embs, entity_atts = [], []
            for entity_num, e in enumerate(entity_pos[i]):
                if len(e) > 1:
                    e_emb, e_att = [], []
                    for start, end in e:
                        if start + offset < c:
                            # In case the entity mention is truncated due to limited max seq length.
                            e_emb.append(sequence_output[i, start + offset])
                            e_att.append(attention[i, :, start + offset])
                    if len(e_emb) > 0:
                        e_emb = torch.logsumexp(torch.stack(e_emb, dim=0), dim=0)
                        e_att = torch.stack(e_att, dim=0).mean(0)
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                else:
                    start, end = e[0]
                    if start + offset < c:
                        e_emb = sequence_output[i, start + offset]
                        e_att = attention[i, :, start + offset]
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                entity_embs.append(e_emb)
                entity_atts.append(e_att)
            for _ in range(self.min_height - entity_num - 1):
                entity_atts.append(e_att)

            entity_embs = torch.stack(entity_embs, dim=0)  # [n_e, d]
            entity_atts = torch.stack(entity_atts, dim=0)  # [n_e, h, seq_len]

            entity_es.append(entity_embs)
            entity_as.append(entity_atts)
            ht_i = torch.LongTensor(hts[i]).to(sequence_output.device)
            hs = torch.index_select(entity_embs, 0, ht_i[:, 0])
            ts = torch.index_select(entity_embs, 0, ht_i[:, 1])

            hss.append(hs)
            tss.append(ts)
        hss = torch.cat(hss, dim=0)
        tss = torch.cat(tss, dim=0)
        return hss, tss, entity_es, entity_as

    def get_enhanced_hrt(self, sequence_output, attention, entity_pos, hts):
        offset = 1 if self.config.transformer_type in ["bert", "roberta"] else 0
        n, h, _, c = attention.size()
        hss, tss, rss = [], [], []
        sent_embs = []
        batch_entity_embs = []
        b, seq_l, h_size = sequence_output.size()
        # n_e = max([len(x) for x in entity_pos])
        n_e = self.min_height
        for i in range(len(entity_pos)):
            entity_embs, entity_atts = [], []
            entity_lens = []

            for e in entity_pos[i]:
                # entity_lens.append(self.ent_num_emb(torch.tensor(len(e)).to(sequence_output).long()))
                if len(e) > 1:
                    e_emb, e_att = [], []
                    for start, end in e:
                        if start + offset < c:
                            # In case the entity mention is truncated due to limited max seq length.
                            # e_emb.append(sequence_output[i, start + offset] + seq_sent_embs[start + offset])
                            e_emb.append(sequence_output[i, start + offset])
                            e_att.append(attention[i, :, start + offset])
                    if len(e_emb) > 0:
                        e_emb = torch.logsumexp(torch.stack(e_emb, dim=0), dim=0)
                        e_att = torch.stack(e_att, dim=0).mean(0)

                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                else:
                    start, end = e[0]
                    if start + offset < c:
                        # e_emb = sequence_output[i, start + offset] + seq_sent_embs[start + offset]
                        e_emb = sequence_output[i, start + offset]
                        e_att = attention[i, :, start + offset]
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                entity_embs.append(e_emb)
                entity_atts.append(e_att)

            entity_embs = torch.stack(entity_embs, dim=0)  # [n_e, d]
            entity_atts = torch.stack(entity_atts, dim=0)  # [n_e, h, seq_len]
            s_ne, _ = entity_embs.size()

            ht_i = torch.LongTensor(hts[i]).to(sequence_output.device)
            hs = torch.index_select(entity_embs, 0, ht_i[:, 0])
            ts = torch.index_select(entity_embs, 0, ht_i[:, 1])
            # print("ht_i.shape: ", ht_i.shape)
            # print("entity_embs.shape: ", entity_embs.shape)
            # print("hs: ", hs)
            # print("ts: ", ts)
            pad_hs = torch.zeros((n_e, n_e, h_size)).to(sequence_output)
            pad_ts = torch.zeros((n_e, n_e, h_size)).to(sequence_output)
            # print("hs.shape: ", hs.shape)  #
            # print("ts.shape: ", ts.shape)
            # print("s_ne: ", s_ne)
            # pad_hs[:s_ne, :s_ne, :] = hs.view(s_ne, s_ne, h_size)  # TODO 这里并不像论文描述那样
            # pad_ts[:s_ne, :s_ne, :] = ts.view(s_ne, s_ne, h_size)
            min_s_ne = min(s_ne, hs.size(0))
            pad_hs[:min_s_ne, :min_s_ne, :] = hs[:min_s_ne, :]
            pad_ts[:min_s_ne, :min_s_ne, :] = ts[:min_s_ne, :]

            h_att = torch.index_select(entity_atts, 0, ht_i[:, 0])
            t_att = torch.index_select(entity_atts, 0, ht_i[:, 1])
            # print(h_att.size())
            # ht_att = (h_att * t_att).mean(1)
            m = torch.nn.Threshold(0, 0)
            ht_att = m((h_att * t_att).sum(1))
            ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-10)

            rs = contract("ld,rl->rd", sequence_output[i], ht_att)
            pad_rs = torch.zeros((n_e, n_e, h_size)).to(sequence_output)
            # pad_rs[:s_ne, :s_ne, :] = rs.view(s_ne, s_ne, h_size)
            pad_rs[:min_s_ne, :min_s_ne, :] = rs[:min_s_ne, :]
            hss.append(pad_hs)
            tss.append(pad_ts)
            rss.append(pad_rs)
            batch_entity_embs.append(entity_embs)
        hss = torch.stack(hss, dim=0)
        tss = torch.stack(tss, dim=0)
        rss = torch.stack(rss, dim=0)  # context
        batch_entity_embs = torch.cat(batch_entity_embs, dim=0)
        return hss, rss, tss, batch_entity_embs

    def get_mask(self, ents, bs, ne, run_device):
        ent_mask = torch.zeros(bs, ne, device=run_device)
        rel_mask = torch.zeros(bs, ne, ne, device=run_device)
        for _b in range(bs):
            ent_mask[_b, : len(ents[_b])] = 1
            rel_mask[_b, : len(ents[_b]), : len(ents[_b])] = 1
        return ent_mask, rel_mask

    def get_ht(self, rel_enco, hts):
        htss = []
        for i in range(len(hts)):
            ht_index = hts[i]
            for h_index, t_index in ht_index:
                htss.append(rel_enco[i, h_index, t_index])
        htss = torch.stack(htss, dim=0)
        return htss

    def get_channel_map(self, sequence_output, entity_as):
        # sequence_output = sequence_output.to('cpu')
        # attention = attention.to('cpu')
        bs, _, d = sequence_output.size()
        # ne = max([len(x) for x in entity_as])  # 本次bs中的最大实体数
        ne = self.min_height

        index_pair = []
        for i in range(ne):
            tmp = torch.cat(
                (torch.ones((ne, 1), dtype=int) * i, torch.arange(0, ne).unsqueeze(1)),
                dim=-1,
            )
            index_pair.append(tmp)
        index_pair = (
            torch.stack(index_pair, dim=0).reshape(-1, 2).to(sequence_output.device)
        )
        map_rss = []
        for b in range(bs):
            entity_atts = entity_as[b]
            h_att = torch.index_select(entity_atts, 0, index_pair[:, 0])
            t_att = torch.index_select(entity_atts, 0, index_pair[:, 1])
            ht_att = (h_att * t_att).mean(1)
            ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-5)
            rs = contract("ld,rl->rd", sequence_output[b], ht_att)
            map_rss.append(rs)
        map_rss = torch.cat(map_rss, dim=0).reshape(bs, ne, ne, d)
        return map_rss

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        entity_pos=None,
        hts=None,
        instance_mask=None,
    ):
        sequence_output, attention = self.encode(
            input_ids, attention_mask, entity_pos
        )  # 用预训练模型编码

        bs, sequen_len, d = sequence_output.shape
        run_device = sequence_output.device.index
        ne = max([len(x) for x in entity_pos])  # 本次bs中的最大实体数
        ent_mask, rel_mask = self.get_mask(entity_pos, bs, ne, run_device)

        # get hs, ts and entity_embs >> entity_rs
        hs, ts, entity_embs, entity_as = self.get_hrt(
            sequence_output, attention, entity_pos, hts
        )

        hs_e, rs_e, ts_e, _ = self.get_enhanced_hrt(
            sequence_output, attention, entity_pos, hts
        )

        # print("hs.shape: ", hs.shape)  # torch.Size([L, 768])
        # print("ts.shape: ", ts.shape)  # torch.Size([L, 768])
        # print("sequence_output.shape: ", sequence_output.shape)  # torch.Size([4, l, 768])
        # 获得通道map的两种不同方法
        if self.channel_type == "context-based":
            feature_map = self.get_channel_map(sequence_output, entity_as)
            # print(
            #     "feature_map:", feature_map.shape
            # )  # torch.Size([4, 42, 42, 768]) DocRED torch.Size([4, 35, 35, 768]) CDR
            attn_input = self.liner(feature_map).permute(0, 3, 1, 2).contiguous()
            # print("attn_input.shape: ", attn_input.shape)

        elif self.channel_type == "grouped-bilinear":
            hs_e = torch.tanh(
                self.enhanced_head_extractor(torch.cat([hs_e, rs_e], dim=3))
            )
            ts_e = torch.tanh(
                self.enhanced_tail_extractor(torch.cat([ts_e, rs_e], dim=3))
            )
            b1_e = hs_e.view(
                bs,
                self.min_height,
                self.min_height,
                self.hidden_size // self.entity_pair_block_size,
                self.entity_pair_block_size,
            )
            b2_e = ts_e.view(
                bs,
                self.min_height,
                self.min_height,
                self.hidden_size // self.entity_pair_block_size,
                self.entity_pair_block_size,
            )

            bl_e = (b1_e.unsqueeze(5) * b2_e.unsqueeze(4)).view(
                bs,
                self.min_height,
                self.min_height,
                self.hidden_size * self.entity_pair_block_size,
            )
            # print("bl_e.shape: ", bl_e.shape)  # torch.Size([4, 35, 49152]) CDR
            attn_input = self.grouped_projection(bl_e)
            # print(
            #     "attn_input.shape: ", attn_input.shape
            # )  # torch.Size([4, 35, 35, 768]) CDR
            attn_input = self.liner(attn_input).permute(0, 3, 1, 2).contiguous()
        elif self.channel_type == "similarity-based":
            ent_encode = sequence_output.new_zeros(bs, self.min_height, d)
            # print(
            #     "ent_encode.shape: ", ent_encode.shape
            # )  # torch.Size([4, 35, 768]) CDR
            for _b in range(bs):
                entity_emb = entity_embs[_b]
                entity_num = entity_emb.size(0)
                ent_encode[_b, :entity_num, :] = entity_emb
            # similar0 = ElementWiseMatrixAttention()(ent_encode, ent_encode).unsqueeze(-1)
            similar1 = DotProductMatrixAttention()(ent_encode, ent_encode).unsqueeze(-1)
            similar2 = CosineMatrixAttention()(ent_encode, ent_encode).unsqueeze(-1)
            similar3 = (
                BilinearMatrixAttention(self.emb_size, self.emb_size)
                .to(ent_encode.device)(ent_encode, ent_encode)
                .unsqueeze(-1)
            )
            attn_input = (
                torch.cat([similar1, similar2, similar3], dim=-1)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            # print("attn_input.shape: ", attn_input.shape)  # torch.Size([4, 3, 35, 35]) CDR
        else:
            raise Exception("channel_type must be specify correctly")

        # print("attn_input.shape:", attn_input.shape)  # (B, 3, 42, 42) DocRED
        # print("attn_input.shape:", attn_input.shape)  # torch.Size([4, 3, 35, 35]) CDR
        # print("attn_input.shape:", attn_input.shape)  # torch.Size([4, 3, 35, 35]) GDA
        attn_map = self.segmentation_net(attn_input)
        # print("attn_map.shape:", attn_map.shape)  # torch.Size([B, 42, 42, 256])  DocRED
        # print("attn_map.shape:", attn_map.shape)  # torch.Size([4, 35, 35, 256])  CDR
        # print("attn_map.shape:", attn_map.shape)  # torch.Size([4, 35, 35, 256]) GDA
        h_t = self.get_ht(attn_map, hts)
        # print("h_t.shape: ", h_t.shape)  # torch.Size([L, 256])
        # print("hs.shape: ", hs.shape)  # torch.Size([L, 768])
        # print("ts.shape: ", ts.shape)  # torch.Size([L, 768])
        # print("hs_e.shape: ", hs_e.shape)  # torch.Size([4, 35, 35, 768])
        # print("h_t.shape: ", h_t.shape)  # torch.Size([L, 256])
        # print("hs.shape: ", hs.shape)  #
        # if self.channel_type == "grouped-bilinear":
        #     hs = torch.tanh(
        #         self.head_extractor(torch.cat([hs_e, h_t], dim=1))
        #     )  # 这里 concat 了, 和 kd 那篇不同，那篇没有 concat
        #     ts = torch.tanh(self.tail_extractor(torch.cat([ts_e, h_t], dim=1)))
        # else:
        hs = torch.tanh(
            self.head_extractor(torch.cat([hs, h_t], dim=1))
        )  # 这里 concat 了, 和 kd 那篇不同，那篇没有 concat
        ts = torch.tanh(self.tail_extractor(torch.cat([ts, h_t], dim=1)))

        b1 = hs.view(-1, self.emb_size // self.block_size, self.block_size)
        b2 = ts.view(-1, self.emb_size // self.block_size, self.block_size)
        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(
            -1, self.emb_size * self.block_size
        )
        logits = self.bilinear(bl)

        output = self.loss_fnt.get_label(logits, num_labels=self.num_labels)
        if labels is not None:
            labels = [torch.tensor(label) for label in labels]
            labels = torch.cat(labels, dim=0).to(logits)
            loss = self.loss_fnt(logits.float(), labels.float())
            output = (loss.to(sequence_output), output)
        return output
