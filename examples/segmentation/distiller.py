import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class Distiller(nn.Module):
    def __init__(self, T=1):
        super().__init__()
        self.T = T

    def distill_logit_loss(self, YS, YT, Label3d, Label_weight):
        if not self.training:
            return None
        Label_weight = Label_weight & (Label3d != 255)

        loss_center, loss_sim = 0., 0.
        for y_s, y_t, label, weight in zip(YS, YT, Label3d, Label_weight):
            y_s, y_t = y_s.flatten(1).permute(1, 0)[weight], y_t.flatten(1).permute(1, 0)[weight]
            label = label[weight]

            # category centers
            unique_label = label.unique()
            unique_label = unique_label[unique_label != 0]
            mask = label[:, None] == unique_label[None, :]
            y_t_center = (y_t[:, None, :] * mask[:, :, None]).sum(0) / mask.sum(0)[:, None]
            y_s_center = (y_s[:, None, :] * mask[:, :, None]).sum(0) / mask.sum(0)[:, None]

            # KLloss for category centers
            p_s = F.log_softmax(y_s_center / self.T, dim=1)
            p_t = F.softmax(y_t_center / self.T, dim=1)
            loss_center = loss_center + (F.kl_div(p_s, p_t, reduction='none') * (self.T ** 2)).sum(-1).mean()

            # MSE loss for relation with category centers
            sim_t = torch.cosine_similarity(y_t[:, None], y_t_center[None, :], dim=-1)
            sim_s = torch.cosine_similarity(y_s[:, None], y_s_center[None, :], dim=-1)
            mseloss = nn.MSELoss()
            loss_sim = loss_sim + mseloss(sim_t, sim_s)
        loss = (loss_center + loss_sim) / Label3d.shape[0]
        return loss

    def distill_feat(self, feat_s, feat_t):
        loss = torch.nn.functional.mse_loss(feat_s, feat_t, reduction='none')
        return loss.mean()

    def forward(self, pred_3d, tsdf_feat, pred_3dT, tsdf_featT, label3d, label_weight):

        loss_distill3d, loss_distillT = 0., 0.

        loss_distill3d = self.distill_logit_loss(pred_3d, pred_3dT, label3d, label_weight)  # label3d,
        for feat_s, feat_t in zip(tsdf_feat, tsdf_featT):
            loss_distillT = loss_distillT + self.distill_feat(feat_s, feat_t)

        return loss_distillT / len(tsdf_feat), loss_distill3d
