import torch.nn as nn
from ..build import MODELS, build_model_from_cfg


@MODELS.register_module()
class VoxelSSC(nn.Module):
    def __init__(self,
                 encoder_args=None,
                 head_args=None,
                 tsdf_args=None,
                 complete_args=None,
                 **kwargs):
        super().__init__()
        self.encoder = build_model_from_cfg(encoder_args)
        self.head = build_model_from_cfg(head_args)

        pretrained = encoder_args.pop('pretrained')
        freeze_encoder = encoder_args.pop('freeze')
        if pretrained:
            self.encoder.init_weights(pretrained)
        if freeze_encoder:
            assert pretrained, 'You freeze the encoder while not loading any pretrained weights'
            self.freeze_encoder()

        self.TSDFNet = build_model_from_cfg(tsdf_args)
        self.complete = build_model_from_cfg(complete_args)

    def freeze_encoder(self):
        for key, param in self.encoder.named_parameters():
            param.requires_grad = False

    def forward(self, img, mapping2d, tsdf=None, **kwargs):
        feature2d = self.encoder(img)
        feature2d, semantic2d = self.head(feature2d)

        tsdf_feat = self.TSDFNet(tsdf)
        pred3d, aug_info = self.complete(feature2d, mapping2d, tsdf_feat)
        return pred3d, semantic2d, tsdf_feat, aug_info