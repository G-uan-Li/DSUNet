import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from model_balanceloss import DocREModel
from utils_sample import set_seed, collate_fn
from prepro import read_cdr, read_gda
import time

# from datetime import datetime


def train(args, model, train_features, dev_features, test_features):
    def logging(s, print_=True, log_=True):
        if print_:
            print(s)
        if log_:
            with open(args.log_dir, "a+") as f_log:
                f_log.write(s + "\n")

    def finetune(features, optimizer, num_epoch, num_steps):
        best_score = -1
        train_dataloader = DataLoader(
            features,
            batch_size=args.train_batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            drop_last=True,
        )
        train_iterator = range(int(num_epoch))   # cdr 125
        # print(len(train_dataloader))
        total_steps = int(
            len(train_dataloader) * num_epoch // args.gradient_accumulation_steps
        )
        print(total_steps)   # 125*30 = 3075
        warmup_steps = int(total_steps * args.warmup_ratio)   # 
        # print(warmup_steps)
        # exit()
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
        print("Total steps: {}".format(total_steps))
        print("Warmup steps: {}".format(warmup_steps))

        log_step = 50
        total_loss = 0
        for epoch in train_iterator:
            start_time = time.time()
            model.zero_grad()
            for step, batch in enumerate(train_dataloader):
                model.train()
                inputs = {
                    "input_ids": batch[0].to(args.device),
                    "attention_mask": batch[1].to(args.device),
                    "labels": batch[2],
                    "entity_pos": batch[3],
                    "hts": batch[4],
                }
                outputs = model(**inputs)
                loss = outputs[0] / args.gradient_accumulation_steps
                loss.backward()
                total_loss += loss.item()
                if step % args.gradient_accumulation_steps == 0:
                    if args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), args.max_grad_norm
                        )

                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                    num_steps += 1
                    if num_steps % log_step == 0:
                        cur_loss = total_loss / log_step
                        elapsed = time.time() - start_time
                        # logging(
                        #    '| epoch {:2d} | step {:4d} | min/b {:5.2f} | lr {} | train loss {:5.3f}'.format(
                        #        epoch, num_steps, elapsed / 60, scheduler.get_lr(), cur_loss * 1000))
                        total_loss = 0
                        start_time = time.time()

                if (step + 1) == len(train_dataloader) - 1 or (
                    args.evaluation_steps > 0
                    and num_steps % args.evaluation_steps == 0
                    and step % args.gradient_accumulation_steps == 0
                ):
                    logging("-" * 89)
                    eval_start_time = time.time()
                    dev_score, dev_output = evaluate(
                        args, model, dev_features, tag="dev"
                    )
                    test_score, test_output = evaluate(
                        args, model, test_features, tag="test"
                    )
                    # print(dev_output)
                    # print(test_output)
                    logging(
                        "| epoch {:3d} | time: {:5.2f}s | dev_output:{} | test_output:{}".format(
                            epoch,
                            time.time() - eval_start_time,
                            dev_output,
                            test_output,
                        )
                    )
                    if test_score > best_score:
                        best_score = test_score
                        logging("best_f1:{}".format(best_score))
                        if args.save_path != "":
                            torch.save(
                                {
                                    "epoch": epoch,
                                    "checkpoint": model.state_dict(),
                                    "best_f1": best_score,
                                },
                                args.save_path,
                            )

        return num_steps

    extract_layer = ["extractor", "bilinear"]
    bert_layer = ["bert_model"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in bert_layer)
            ],
            "lr": args.bert_lr,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in extract_layer)
            ],
            "lr": 1e-4,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in extract_layer + bert_layer)
            ]
        },
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )   # 比较集中优化器，还是awamd好用
    num_steps = 0
    set_seed(args)
    model.zero_grad()
    finetune(train_features, optimizer, args.num_train_epochs, num_steps)


def evaluate(args, model, features, tag="dev"):
    dataloader = DataLoader(
        features,
        batch_size=args.test_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
    )
    preds, golds = [], []
    for i, batch in enumerate(dataloader):
        model.eval()

        inputs = {
            "input_ids": batch[0].to(args.device),
            "attention_mask": batch[1].to(args.device),
            "labels": batch[2],
            "entity_pos": batch[3],
            "hts": batch[4],
        }

        with torch.no_grad():
            output = model(**inputs)
            loss = output[0]
            pred = output[1].cpu().numpy()
            pred[np.isnan(pred)] = 0
            preds.append(pred)
            golds.append(
                np.concatenate(
                    [np.array(label, np.float32) for label in batch[2]], axis=0
                )
            )

    preds = np.concatenate(preds, axis=0).astype(np.float32)
    golds = np.concatenate(golds, axis=0).astype(np.float32)
    tp = ((preds[:, 1] == 1) & (golds[:, 1] == 1)).astype(np.float32).sum()
    tn = ((golds[:, 1] == 1) & (preds[:, 1] != 1)).astype(np.float32).sum()
    fp = ((preds[:, 1] == 1) & (golds[:, 1] != 1)).astype(np.float32).sum()
    precision = tp / (tp + fp + 1e-5)
    recall = tp / (tp + tn + 1e-5)
    f1 = 2 * precision * recall / (precision + recall + 1e-5)
    output = {
        tag + "_F1": f1 * 100,
        tag + "_P": precision * 100,
        tag + "_R": recall * 100,
    }
    return f1, output


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="./dataset/cdr", type=str)
    parser.add_argument("--transformer_type", default="bert", type=str)
    parser.add_argument(
        "--model_name_or_path", default="allenai/scibert_scivocab_cased", type=str
    )

    parser.add_argument("--train_file", default="train_filter.data", type=str)
    parser.add_argument("--dev_file", default="dev_filter.data", type=str)
    parser.add_argument("--test_file", default="test_filter.data", type=str)
    parser.add_argument(
        "--entity_pair_block_size",
        default=64,
        type=int,
        help="block_size for entity pair representation, from document-level relation extraction with adaptive focal loss and kd",
    )
    parser.add_argument("--save_path", default="", type=str)
    parser.add_argument("--load_path", default="", type=str)

    parser.add_argument(
        "--config_name",
        default="",
        type=str,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--max_seq_length",
        default=1024,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )

    parser.add_argument(
        "--train_batch_size", default=4, type=int, help="Batch size for training."
    )
    parser.add_argument(
        "--test_batch_size", default=8, type=int, help="Batch size for testing."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        default=1,
        type=int,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--num_labels",
        default=1,
        type=int,
        help="Max number of labels in the prediction.",
    )
    parser.add_argument(
        "--learning_rate",
        default=2e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--warmup_ratio", default=0.06, type=float, help="Warm up ratio for Adam."
    )
    parser.add_argument(
        "--num_train_epochs",
        default=30.0,
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--evaluation_steps",
        default=-1,
        type=int,
        help="Number of training steps between evaluations.",
    )
    parser.add_argument(
        "--seed", type=int, default=66, help="random seed for initialization."
    )
    parser.add_argument(
        "--num_class", type=int, default=2, help="Number of relation types in dataset."
    )
    # UNet parameters
    parser.add_argument("--unet_in_dim", type=int, default=3, help="unet_in_dim.")
    parser.add_argument("--unet_out_dim", type=int, default=256, help="unet_out_dim.")
    parser.add_argument("--down_dim", type=int, default=256, help="down_dim.")
    parser.add_argument("--channel_type", type=str, default="", help="unet_out_dim.")
    parser.add_argument("--log_dir", type=str, default="", help="log.")
    parser.add_argument(
        "--bert_lr",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument("--max_height", type=int, default=42, help="log.")

    # SwinTransformer parameters
    parser.add_argument(
        "--is_using_swin", type=bool, default=True, help="是否用 SwinUnet 代替 CNNUnet."
    )
    parser.add_argument(
        "--IMG_SIZE",
        type=int,
        default=48,
        help="关系矩阵的边长. 48 for DocRED, 48 for CDR, 48 for GDA",
    )
    parser.add_argument(
        "--DEPTHS",
        type=list,
        default=[2, 2],
        help="几个 stage，每个 stage 几层（每个 stage 必须是偶数层）",
    )
    parser.add_argument("--PATCH_SIZE", type=int, default=2, help="patch_size.")
    parser.add_argument("--WINDOW_SIZE", type=int, default=4, help="window_size.")
    parser.add_argument(
        "--NUM_HEADS", type=list, default=[3, 6], help="每个 stage 中 MHSA 的num_heads."
    )
    parser.add_argument(
        "--EMBED_DIM",
        type=int,
        default=48,
        help="embed_dim. [96, 48]也就是 swin paper 中的C",
    )

    # 一般不需要调节的超参数
    parser.add_argument("--IN_CHANS", type=int, default=3, help="in_chans 通道数.")
    parser.add_argument("--CLASS_NUMBER", type=int, default=256, help="class_number.")

    # parser.add_argument("--DEPTHS_DECODER", type=list, default=[1, 2], help="")
    # depths_decoder
    parser.add_argument("--DROP_RATE", type=float, default=0.05, help="drop_rate.")
    parser.add_argument(
        "--DROP_PATH_RATE", type=float, default=0.0, help="drop_path_rate."
    )
    parser.add_argument("--APE", type=bool, default=True, help="ape. 是否用位置编码")
    parser.add_argument(
        "--PATCH_NORM",
        type=bool,
        default=True,
        help="If True, add normalization after patch embedding. Default: True",
    )

    args = parser.parse_args()
    # wandb.init(project="CDR")
    print(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=args.num_class,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    )
    # print(type(tokenizer))
    print(len(tokenizer.vocab))

    read = read_cdr if "cdr" in args.data_dir else read_gda

    train_file = os.path.join(args.data_dir, args.train_file)
    dev_file = os.path.join(args.data_dir, args.dev_file)
    test_file = os.path.join(args.data_dir, args.test_file)
    if "cdr" in args.data_dir:
        train_cacae_file_name = "./dataset/cdr/cache/train_cache_cdr"
        dev_cache_file_name = "./dataset/cdr/cache/dev_cache_cdr"
        test_cache_file_name = "./dataset/cdr/cache/test_cache_cdr"
    elif "Bio_ClinicalBERT" in args.model_name_or_path:
        train_cacae_file_name = "./dataset/gda/cache/train_Bio_ClinicalBERT_cache_gda"
        dev_cache_file_name = "./dataset/gda/cache/dev_Bio_ClinicalBERT_cache_gda"
        test_cache_file_name = "./dataset/gda/cache/test_Bio_ClinicalBERT_cache_gda"
    elif "biobert-large" in args.model_name_or_path:
        train_cacae_file_name = "./dataset/gda/cache/train_biobert-large_cache_gda"
        dev_cache_file_name = "./dataset/gda/cache/dev_biobert-large_cache_gda"
        test_cache_file_name = "./dataset/gda/cache/test_biobert-large_cache_gda"
    elif "biobert" in args.model_name_or_path:
        train_cacae_file_name = "./dataset/gda/cache/train_biobert_cache_gda"
        dev_cache_file_name = "./dataset/gda/cache/dev_biobert_cache_gda"
        test_cache_file_name = "./dataset/gda/cache/test_biobert_cache_gda"
    else:
        train_cacae_file_name = "./dataset/gda/cache/train_cache_gda"
        dev_cache_file_name = "./dataset/gda/cache/dev_cache_gda"
        test_cache_file_name = "./dataset/gda/cache/test_cache_gda"

    train_features = read(
        train_file, train_cacae_file_name, tokenizer, max_seq_length=args.max_seq_length
    )
    dev_features = read(
        dev_file, dev_cache_file_name, tokenizer, max_seq_length=args.max_seq_length
    )
    test_features = read(
        test_file, test_cache_file_name, tokenizer, max_seq_length=args.max_seq_length
    )

    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )

    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    config.transformer_type = args.transformer_type

    set_seed(args)
    model = DocREModel(
        config,
        args,
        model,
        block_size=args.entity_pair_block_size,
        num_labels=args.num_labels,
    )
    model.to(0)

    if args.load_path == "":
        train(args, model, train_features, dev_features, test_features)
    else:
        # model = amp.initialize(model, opt_level="O1", verbosity=0)
        model.load_state_dict(torch.load(args.load_path))
        dev_score, dev_output = evaluate(args, model, dev_features, tag="dev")
        test_score, test_output = evaluate(args, model, test_features, tag="test")
        print(dev_output)
        print(test_output)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
