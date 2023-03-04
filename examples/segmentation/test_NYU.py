import json

import __init__
import argparse
import yaml
import os
import logging
import numpy as np
import glob
import pathlib
import csv
from cleaner.models import build_model_from_cfg
from cleaner.utils import set_random_seed,  load_checkpoint, setup_logger_dist, \
    cal_model_parm_nums, generate_exp_directory, EasyConfig, dist_utils
import warnings
import torch
from cleaner.utils import ConfusionMatrix, get_mious
from tqdm import tqdm
from cleaner.dataset import build_dataloader_from_cfg
from terminaltables import AsciiTable

warnings.simplefilter(action='ignore', category=FutureWarning)


def get_xyz(size):
    """x 水平 y高低  z深度"""
    _x = np.zeros(size, dtype=np.int32)
    _y = np.zeros(size, dtype=np.int32)
    _z = np.zeros(size, dtype=np.int32)

    for i_h in range(size[0]):  # x, y, z
        _x[i_h, :, :] = i_h  # x, left-right flip
    for i_w in range(size[1]):
        _y[:, i_w, :] = i_w  # y, up-down flip
    for i_d in range(size[2]):
        _z[:, :, i_d] = i_d  # z, front-back flip
    return _x, _y, _z


def labeled_voxel2ply(vox_labeled, ply_filename, colorMap=None):  #
    """Save labeled voxels to disk in colored-point cloud format: x y z r g b, with '.ply' suffix
           vox_labeled.shape: (W, H, D)
        """  #
    if colorMap is None:
        colorMap = np.array([[22, 191, 206],  # 0 empty, free space
                             [214, 38, 40],  # 1 ceiling
                             [43, 160, 4],  # 2 floor
                             [158, 216, 229],  # 3 wall
                             [114, 158, 206],  # 4 window
                             [204, 204, 91],  # 5 chair  new: 180, 220, 90
                             [255, 186, 119],  # 6 bed
                             [147, 102, 188],  # 7 sofa
                             [30, 119, 181],  # 8 table
                             [188, 188, 33],  # 9 tvs
                             [255, 127, 12],  # 10 furn
                             [196, 175, 214],  # 11 objects
                             [153, 153, 153],  # 12 Accessible area, or label==255, ignore
                             ]).astype(np.int32)

    # ---- Check data type, numpy ndarray
    if type(vox_labeled) is not np.ndarray:
        raise Exception(
            "Oops! Type of vox_labeled should be 'numpy.ndarray', not {}.".
                format(type(vox_labeled)))
    # ---- Check data validation
    if np.amax(vox_labeled) == 0:
        print('Oops! All voxel is labeled empty.')
        return
    # ---- get size
    size = vox_labeled.shape
    # ---- Convert to list
    vox_labeled = vox_labeled.flatten()
    # ---- Get X Y Z
    _x, _y, _z = get_xyz(size)
    _x = _x.flatten()
    _y = _y.flatten()
    _z = _z.flatten()
    # ---- Get R G B
    vox_labeled[vox_labeled == 255] = 0  # empty
    _rgb = colorMap[vox_labeled[:]]
    xyz_rgb = zip(_x, _y, _z, _rgb[:, 0], _rgb[:, 1], _rgb[:, 2])
    xyz_rgb = list(xyz_rgb)

    xyz_rgb = np.array(xyz_rgb)
    ply_data = xyz_rgb[np.where(vox_labeled > 0)]

    if len(ply_data) == 0:
        raise Exception("Oops!  That was no valid ply data.")
    ply_head = 'ply\n' \
               'format ascii 1.0\n' \
               'element vertex %d\n' \
               'property float x\n' \
               'property float y\n' \
               'property float z\n' \
               'property uchar red\n' \
               'property uchar green\n' \
               'property uchar blue\n' \
               'end_header' % len(ply_data)
    # ---- Save ply data to disk
    np.savetxt(ply_filename,
               ply_data,
               fmt="%d %d %d %d %d %d",
               header=ply_head,
               comments='')
    del vox_labeled, _x, _y, _z, _rgb, xyz_rgb, ply_data, ply_head
    print('Saved-->{}'.format(ply_filename))


def visualize_3d_predict(pred3d, label_weight, file, visualize,
                         visual_result='../../visual_pred/CleanerS'):
    if not os.path.exists(visual_result):
        os.makedirs(visual_result)

    pred = pred3d.view(60, 36, 60).cpu().numpy()
    label_weight = label_weight.view(60, 36, 60).cpu().numpy()
    pred[label_weight == 0] = 0
    if visualize:
        labeled_voxel2ply(pred, ply_filename='{}/{}.ply'.format(visual_result, file[0]))


def format_print(ious, accs, recs, mean_index=0, print_eval=False):
    CLASSES = ('SceneComp', 'ceiling', 'floor', 'wall', 'window', 'chair', 'bed', 'sofa', 'table', 'tvs', 'furn', 'objects')
    label2cat = {
        i: cat_name
        for i, cat_name in enumerate(CLASSES)
    }
    header = ['classes']
    for i in range(len(label2cat)):
        header.append(label2cat[i])
    header.extend(['mean'])

    ret_dict = dict()
    table_columns, table_accs, table_recs = [['iou']], [['accuracy']], [['recall']]
    for i in range(len(label2cat)):
        ret_dict[label2cat[i]] = float(ious[i])
        table_columns.append([f'{ious[i]:.4f}'])
        table_accs.append([f'{accs[i]:.4f}'])
        table_recs.append([f'{recs[i]:.4f}'])
    miou, macc, mrec = np.mean(ious[mean_index:]), np.mean(accs[mean_index:]), np.mean(recs[mean_index:])
    table_columns.append([f'{miou:.4f}'])
    table_accs.append([f'{macc:.4f}'])
    table_recs.append([f'{mrec:.4f}'])

    table_data = [header]
    table_data += list(zip(*table_accs))
    table_data += list(zip(*table_recs))
    table_data += list(zip(*table_columns))
    table = AsciiTable(table_data)
    table.inner_footing_row_border = True
    if print_eval:
        print('\n' + table.table)
    return miou


def write_to_csv(SC_acc, SC_rec, SC_miou, SSC_ious, save_name, csv_path, write_header=True):
    ious_table = [f'{item:.1f}' for item in SSC_ious]
    header = ['method', 'prec.', 'recall', 'IoU',] + cfg.classes[1:] + ['avg']
    data = [save_name, f'{SC_acc:.1f}', f'{SC_rec:.1f}', f'{SC_miou:.1f}'] + ious_table + [f'{np.mean(SSC_ious):.1f}']
    with open(csv_path, 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(data)
        f.close()


def format_SSC_eval(cmSC, cmSSC, description=None, print_eval=False):
    if description is not None:
        print(description)

    SC_miou, SC_macc, SC_oa, SC_mrec, SC_ious, SC_accs, SC_recs = cmSC.all_metrics(with_recall=True)
    SC_miou, SC_acc, SC_rec = SC_ious[1], SC_accs[1], SC_recs[1]
    SSC_miou, SSC_macc, SSC_oa, SSC_mrec, SSC_ious, SSC_accs, SSC_recs = cmSSC.all_metrics(with_recall=True)
    SSC_ious[0], SSC_accs[0], SSC_recs[0] = SC_miou, SC_acc, SC_rec
    SSC_miou = format_print(SSC_ious, SSC_accs, SSC_recs, mean_index=1, print_eval=print_eval)
    return SC_miou, SSC_miou, SSC_ious


def test(cfg, visualize=True):
    model = build_model_from_cfg(cfg.model).to(cfg.rank)
    model_size = cal_model_parm_nums(model)
    logging.info('Number of params: %.4f M' % (model_size / 1e6))
    load_checkpoint(model, pretrained_path=cfg.pretrained_path, prefix='student.')

    model.eval()  # set model to eval mode
    cmSC_ = ConfusionMatrix(num_classes=cfg.num_classes, ignore_index=cfg.ignore_index)
    cmSSC_ = ConfusionMatrix(num_classes=cfg.num_classes, ignore_index=cfg.ignore_index)
    pbar = tqdm(enumerate(val_loader), total=val_loader.__len__())

    eval_res = {}
    for idx, data in pbar:
        keys = data.keys() if callable(data.keys) else data.keys
        for key in keys:
            if key == 'file':
                continue
            data[key] = data[key].cuda(non_blocking=True)

        pred_3d, _, _, _ = model(**data)

        label_weight, label3d, mapping = data['label_weight'], data['label3d'], data['mapping']

        pred_3d = pred_3d.flatten(2).permute(0, 2, 1)

        pred_3dSSC = pred_3d.argmax(dim=2)

        label_weightSSC = label_weight & (label3d != cfg.ignore_index)
        pred_3dSSC, targetSSC = pred_3dSSC[label_weightSSC], label3d[label_weightSSC]
        cmSSC_.update(pred_3dSSC, targetSSC)

        weightSC = label_weight & (mapping == 307200) & (label3d != cfg.ignore_index)
        pred_3dSC, targetSC = (pred_3d[weightSC].argmax(dim=1) > 0).long(), (label3d[weightSC] > 0).long()
        cmSC_.update(pred_3dSC, targetSC)
        visualize_3d_predict(pred_3d[0].argmax(dim=1), label_weight[0], data['file'], visualize=visualize)
    print(cmSSC_.value)
    format_SSC_eval(cmSC_, cmSSC_, print_eval=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('semantic scene segmentation test')
    parser.add_argument('--cfg', type=str, default='../../cfgs/NYU/voxelSSC.yaml', help='config file')
    args, opts = parser.parse_known_args()
    cfg = EasyConfig()
    cfg.load(args.cfg, recursive=True)
    cfg.update(opts)

    if cfg.seed is None:
        cfg.seed = np.random.randint(1, 10000)
    # init distributed env first, since logger depends on the dist info.
    cfg.rank, cfg.world_size, cfg.distributed, cfg.mp = dist_utils.get_dist_info(cfg)

    cfg.mode = 'test'
    # init log dir
    cfg.task_name = args.cfg.split('.')[-2].split('/')[-2]
    cfg.exp_name = args.cfg.split('.')[-2].split('/')[-1]
    tags = [
        cfg.task_name,  # task name (the folder of name under ./cfgs
        cfg.mode,
        cfg.exp_name,  # cfg file name
    ]
    for i, opt in enumerate(opts):
        if 'rank' not in opt and 'dir' not in opt and 'root' not in opt and 'path' not in opt and 'wandb' not in opt and '/' not in opt:
            tags.append(opt)
    cfg.root_dir = os.path.join(cfg.root_dir, cfg.task_name)

    generate_exp_directory(
        cfg, tags, additional_id=os.environ.get('MASTER_PORT', None))

    os.environ["JOB_LOG_DIR"] = cfg.log_dir
    cfg_path = os.path.join(cfg.run_dir, "cfg.yaml")
    with open(cfg_path, 'w') as f:
        yaml.dump(cfg, f, indent=2)
        os.system('cp %s %s' % (args.cfg, cfg.run_dir))
    cfg.cfg_path = cfg_path

    # logger
    setup_logger_dist(cfg.log_path, cfg.rank, name=cfg.dataset.common.NAME)

    # build dataset
    val_loader = build_dataloader_from_cfg(cfg.get('val_batch_size', cfg.batch_size),
                                           cfg.dataset,
                                           cfg.dataloader,
                                           datatransforms_cfg=cfg.datatransforms,
                                           split='val',
                                           distributed=False
                                           )
    cfg.classes = val_loader.dataset.classes
    test(cfg)
