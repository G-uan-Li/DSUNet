import logging
import torch.nn as nn
import torch
from swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys

logger = logging.getLogger(__name__)


class SwinUnet(nn.Module):
    def __init__(self, config):
        super(SwinUnet, self).__init__()
        self.config = config

        self.swin_unet = SwinTransformerSys(img_size=config.IMG_SIZE, patch_size=config.PATCH_SIZE, in_chans=config.IN_CHANS, class_number=config.CLASS_NUMBER, embed_dim=config.EMBED_DIM, depths=config.DEPTHS, num_heads=config.NUM_HEADS, window_size=config.WINDOW_SIZE, drop_rate=config.DROP_RATE, drop_path_rate=config.DROP_PATH_RATE, ape=config.APE, patch_norm=config.PATCH_NORM)

    def forward(self, x):
        if x.size()[1] == 1:  # 如果输入是单通道的，就复制成三通道
            x = x.repeat(1, 3, 1, 1)
        # torch.Size([4, 3, 42, 42])
        x_padded = torch.zeros(x.size(0), x.size(1), self.config.IMG_SIZE, self.config.IMG_SIZE).to(x.device)
        x_padded[:, :, : x.size(2), : x.size(3)] = x

        # x = torch.cat([x, torch.zeros(x.size(0), x.size(1), 6, x.size(3))], 2).to(x.device)
        # x = torch.cat([x, torch.zeros(x.size(0), x.size(1), x.size(2), 6)], 3).to(x.device)
        # torch.Size([4, 3, 48, 48])
        # print("Swin Unet input shape: ", x_padded.shape)
        logits = self.swin_unet(x_padded)
        logits = logits[:, :, : x.size(2), : x.size(3)]
        # print("after Swin Unet shape: ", logits.shape)  # torch.Size([4, 256, 42, 42])
        logits = logits.permute(0, 2, 3, 1).contiguous()
        return logits
