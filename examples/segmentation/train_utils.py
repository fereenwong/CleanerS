import argparse, yaml, os, logging, numpy as np, csv
from tqdm import tqdm
import torch, torch.nn as nn
from torch import distributed as dist, multiprocessing as mp
# from torch_scatter import scatter
from cleaner.utils import set_random_seed, load_checkpoint, save_checkpoint, cal_model_parm_nums
from cleaner.utils import AverageMeter, ConfusionMatrix, get_mious
from cleaner.dataset import build_dataloader_from_cfg
from cleaner.transforms import build_transforms_from_cfg
from cleaner.optim import build_optimizer_from_cfg
from cleaner.scheduler import build_scheduler_from_cfg
from cleaner.loss import build_criterion_from_cfg
from cleaner.models import build_model_from_cfg
from distiller import Distiller


def model_info(model, report='full'):
    # Plots a line-by-line description of a PyTorch model
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    count_enc, count_dec = 0, 0
    if report is 'full':
        print('%5s %80s %9s %12s %20s %10s %10s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            if 'backbone' in name:
                count_enc += p.numel()
            else:
                count_dec += p.numel()
            print('%5g %80s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
    print('Model Summary: %g layers, %g parameters, %g gradients' % (len(list(model.parameters())), n_p, n_g))
    print('Model summary: %g parameters in Encoder, %g parameters in Decoder' % (count_enc, count_dec))


def train_one_epoch(model, train_loader, optimizer, scheduler, epoch, cfg, mode='T', teacher=None):
    loss2d_meter, loss3d_meter, lossdis_meter = AverageMeter(), AverageMeter(), AverageMeter()
    cm = ConfusionMatrix(num_classes=cfg.num_classes, ignore_index=cfg.ignore_index)
    cmSC = ConfusionMatrix(num_classes=cfg.num_classes, ignore_index=cfg.ignore_index)
    cmSSC = ConfusionMatrix(num_classes=cfg.num_classes, ignore_index=cfg.ignore_index)
    distiller = Distiller()

    model.train()  # set model to training mode
    pbar = tqdm(enumerate(train_loader), ncols=100, total=train_loader.__len__())
    num_iter = 0
    for idx, data in pbar:
        keys = data.keys() if callable(data.keys) else data.keys
        for key in keys:
            if isinstance(data[key], str) or isinstance(data[key], list):
                continue
            data[key] = data[key].cuda(non_blocking=True)
        num_iter += 1

        img, mapping2d, tsdf = data['img'], data['mapping2d'], data['tsdf_CAD' if mode == 'T' else 'tsdf']
        pred_3d, pred_2d, tsdf_feat, aug_info = model(img, mapping2d, tsdf)

        if aug_info['flip_z']:
            pred_3d = pred_3d.flip([4, ])
        if aug_info['flip_x']:
            pred_3d = pred_3d.flip([2, ])

        label_weight, label3d, mapping, mapping2d = data['label_weight'], data['label3d'], \
                                                    data['mapping'], data['mapping2d']

        if teacher is not None and mode == 'S':
            tsdf_CAD = data['tsdf_CAD']

            pred_3dT, pred_2dT, tsdf_featT, _ = teacher(img, mapping2d, tsdf_CAD)
            loss_KDT, loss_KDS = distiller(pred_3d, tsdf_feat, pred_3dT, tsdf_featT, label3d, label_weight)

        mapping2d = mapping2d.flatten(1)
        criterion = build_criterion_from_cfg(cfg.criterion).cuda()
        pred_2d = pred_2d.flatten(2).permute(0, 2, 1)

        pred_2d = torch.cat([pred_2d[i][mapping2d[i] != -1] for i in range(len(pred_2d))])
        label2d = torch.cat([label3d[i][mapping2d[i][mapping2d[i] != -1]] for i in range(len(label3d))])
        loss2d = criterion(pred_2d, label2d)

        '''
        loss 3d
        '''
        pred_3d = pred_3d.flatten(2).permute(0, 2, 1)

        weightSSC = label_weight & (label3d != cfg.ignore_index)
        pred_3dSSC, label3dSSC = pred_3d[weightSSC], label3d[weightSSC]
        loss3d = criterion(pred_3dSSC, label3dSSC)

        loss = loss2d * 0.25 + loss3d
        if mode == 'S' and teacher is not None:
            loss = loss + 0.25 * (loss_KDT + loss_KDS)

        loss.backward()
        # optimize
        if num_iter == cfg.step_per_update:
            if cfg.get('grad_norm_clip') is not None and cfg.grad_norm_clip > 0.:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_norm_clip, norm_type=2)
            num_iter = 0
            optimizer.step()
            optimizer.zero_grad()
            if not cfg.sched_on_epoch:
                scheduler.step(epoch)

        # update confusion matrix
        weightSC = label_weight & (mapping == 307200) & (label3d != cfg.ignore_index)
        pred_3dSC, label3dSC = pred_3d[weightSC], label3d[weightSC]

        cm.update(pred_2d.argmax(dim=1), label2d)
        pred_3dSC, label3dSC = (pred_3dSC.argmax(dim=1) > 0).long(), (label3dSC > 0).long()
        cmSC.update(pred_3dSC, label3dSC)
        cmSSC.update(pred_3dSSC.argmax(dim=1), label3dSSC)
        loss2d_meter.update(loss2d.item())
        loss3d_meter.update(loss3d.item())

        if idx % cfg.print_freq:
            if mode == 'T':
                pbar.set_description(f"Train teacher Epoch [{epoch}/{cfg.epochs}] "
                                     f"Loss2d {loss2d_meter.val:.3f} Loss3d {loss3d_meter.val:.3f} "
                                     f"Acc {cmSSC.overall_accuray:.2f}")
            elif mode == 'S' and teacher is not None:
                lossdis_meter.update((loss_KDT + loss_KDS).item())
                pbar.set_description(f"Train student Epoch [{epoch}/{cfg.epochs}] "
                                     f"Loss2d {loss2d_meter.val:.3f} Loss3d {loss3d_meter.val:.3f} "
                                     f"Lossdis {lossdis_meter.val: .3f} "
                                     f"Acc {cmSSC.overall_accuray:.2f}")
            else:
                pbar.set_description(f"Train Epoch [{epoch}/{cfg.epochs}] "
                                     f"Loss2d {loss2d_meter.val:.3f} Loss3d {loss3d_meter.val:.3f} "
                                     f"Acc {cmSSC.overall_accuray:.2f}")
    miou, macc, oa, ious, accs = cm.all_metrics()
    miouSC, maccSC, oaSC, iousSC, accsSC = cmSC.all_metrics()
    miouSC = iousSC[1]
    miouSSC, maccSSC, oaSSC, iousSSC, accsSSC = cmSSC.all_metrics()
    miouSSC = np.mean(iousSSC[1:])
    return loss2d_meter.avg, loss3d_meter.avg, miou, miouSC, miouSSC, iousSSC


@torch.no_grad()
def validate(model, val_loader, cfg, num_votes=1, data_transform=None, mode='T'):
    model.eval()  # set model to eval mode
    cm = ConfusionMatrix(num_classes=cfg.num_classes, ignore_index=cfg.ignore_index)
    cmSC = ConfusionMatrix(num_classes=cfg.num_classes, ignore_index=cfg.ignore_index)
    cmSSC = ConfusionMatrix(num_classes=cfg.num_classes, ignore_index=cfg.ignore_index)
    pbar = tqdm(enumerate(val_loader), total=val_loader.__len__())
    for idx, data in pbar:
        keys = data.keys() if callable(data.keys) else data.keys
        for key in keys:
            if isinstance(data[key], str) or isinstance(data[key], list):
                continue
            data[key] = data[key].cuda(non_blocking=True)

        img, mapping2d, tsdf = data['img'], data['mapping2d'], data['tsdf_CAD' if mode == 'T' else 'tsdf']
        pred_3d, pred_2d, _, _ = model(img, mapping2d, tsdf)

        label_weight, label3d, mapping = data['label_weight'], data['label3d'], data['mapping']
        pred_2d = pred_2d.flatten(2).permute(0, 2, 1)
        pred_2d = torch.cat([pred_2d[i][mapping[i][mapping[i] != 307200]] for i in range(len(pred_2d))])
        label2d = torch.cat([label3d[i][mapping[i] != 307200] for i in range(len(label3d))])

        cm.update(pred_2d.argmax(dim=1), label2d)

        pred_3d = pred_3d.flatten(2).permute(0, 2, 1)
        weightSSC = label_weight & (label3d != cfg.ignore_index)
        pred_3dSSC, targetSSC = pred_3d[weightSSC], label3d[weightSSC]
        cmSSC.update(pred_3dSSC.argmax(dim=1), targetSSC)

        weightSC = label_weight & (data['mapping'] == 307200) & (label3d != cfg.ignore_index)
        pred_3dSC, targetSC = (pred_3d[weightSC].argmax(dim=1) > 0).long(), (label3d[weightSC] > 0).long()
        cmSC.update(pred_3dSC, targetSC)

    tp, union, count = cm.tp, cm.union, cm.count
    tpSC, unionSC, countSC = cmSC.tp, cmSC.union, cmSC.count
    tpSSC, unionSSC, countSSC = cmSSC.tp, cmSSC.union, cmSSC.count

    if cfg.distributed:
        dist.all_reduce(tp), dist.all_reduce(union), dist.all_reduce(count)
        dist.all_reduce(tpSC), dist.all_reduce(unionSC), dist.all_reduce(countSC)
        dist.all_reduce(tpSSC), dist.all_reduce(unionSSC), dist.all_reduce(countSSC)
    # print(f'after gathering, rank {cfg.rank} \n union {union} \n count {count}')
    miou, macc, oa, ious, accs = get_mious(tp, union, count)
    miouSC, maccSC, oaSC, iousSC, accsSC = get_mious(tpSC, unionSC, countSC)
    miouSC = iousSC[1]
    miouSSC, maccSSC, oaSSC, iousSSC, accsSSC = get_mious(tpSSC, unionSSC, countSSC)
    miouSSC = np.mean(iousSSC[1:])
    return miou, ious, miouSC, miouSSC, iousSSC, accsSSC


def train_teacher(gpu, cfg):
    teacher = build_model_from_cfg(cfg.model).to(cfg.rank)
    model_size = cal_model_parm_nums(teacher)
    logging.info(teacher)
    logging.info('Number of params: %.4f M' % (model_size / 1e6))

    if cfg.sync_bn:
        teacher = torch.nn.SyncBatchNorm.convert_sync_batchnorm(teacher)
        logging.info('Using Synchronized BatchNorm ...')
    if cfg.distributed:
        torch.cuda.set_device(gpu)
        teacher = nn.parallel.DistributedDataParallel(teacher.cuda(), device_ids=[cfg.rank], output_device=cfg.rank,
                                                    find_unused_parameters=True)
        logging.info('Using Distributed Data parallel ...')

    # optimizer & scheduler
    optimizer = build_optimizer_from_cfg(teacher, lr=cfg.lr, **cfg.optimizer)
    scheduler = build_scheduler_from_cfg(cfg, optimizer)

    # build dataset
    val_loader = build_dataloader_from_cfg(cfg.get('val_batch_size', cfg.batch_size),
                                           cfg.dataset,
                                           cfg.dataloader,
                                           datatransforms_cfg=cfg.datatransforms,
                                           split='val',
                                           distributed=cfg.distributed
                                           )
    logging.info(f"length of validation dataset: {len(val_loader.dataset)}")

    train_loader = build_dataloader_from_cfg(cfg.batch_size,
                                             cfg.dataset,
                                             cfg.dataloader,
                                             datatransforms_cfg=cfg.datatransforms,
                                             split='train',
                                             distributed=cfg.distributed,
                                             )
    logging.info(f"length of training dataset: {len(train_loader.dataset)}")

    # ===> start training
    val_miou, val_macc, val_oa, val_ious, val_accs = 0., 0., 0., [], []
    best_val, macc_when_best, oa_when_best, ious_when_best, best_epoch = 0., 0., 0., [], 0
    for epoch in range(cfg.start_epoch, cfg.epochs + 1):
        if cfg.distributed:
            train_loader.sampler.set_epoch(epoch)
        if hasattr(train_loader.dataset, 'epoch'):  # some dataset sets the dataset length as a fixed steps.
            train_loader.dataset.epoch = epoch - 1
        train2d_loss, train3d_loss, train_miou2d, train_miouSC, train_miouSSC, train_iousSSC = \
            train_one_epoch(teacher, train_loader, optimizer, scheduler, epoch, cfg)

        is_best = False
        cfg.val_freq = 1
        if epoch % cfg.val_freq == 0:
            val_miou2d, val_ious2d, val_miouSC, val_miouSSC, val_iousSSC, val_accsSSC = validate(teacher, val_loader,
                                                                                                    cfg)
            if val_miouSSC > best_val:
                is_best = True
                best_val = val_miouSSC
                macc_when_best = val_macc
                oa_when_best = val_oa
                ious_when_best = val_ious
                best_epoch = epoch
                with np.printoptions(precision=2, suppress=True):
                    logging.info(
                        f'Find a better ckpt @E{epoch}, val_miou2d {val_miou2d:.2f} \nval_ious2d: {val_ious2d}'
                        f'\nvalSC_miou {val_miouSC:.2f} valSSC_miou {val_miouSSC:.2f}'
                        f'\nval_iousSSC: {val_iousSSC}')

        lr = optimizer.param_groups[0]['lr']
        logging.info(f'Epoch {epoch} LR {lr:.6f} '
                     f'train_miou {train_miou2d:.2f}, train_miouSC {train_miouSC:.2f}, '
                     f'train_miouSSC {train_miouSSC:.2f},'
                     f'val_miou {val_miou2d:.2f}, val_miouSC {val_miouSC:.2f}, '
                     f'val_miouSSC {val_miouSSC:.2f}, best val SSC miou {best_val:.2f}')

        if cfg.sched_on_epoch:
            scheduler.step(epoch)
        if cfg.rank == 0:
            cfg.ckpt_dir = './teacher/'
            if not os.path.exists(cfg.ckpt_dir):
                os.makedirs(cfg.ckpt_dir)
            save_checkpoint(cfg, teacher, epoch, optimizer, scheduler,
                            additioanl_dict={'best_val': best_val},
                            is_best=is_best, save_name='Teacher'
                            )

    # validate
    with np.printoptions(precision=2, suppress=True):
        logging.info(
            f'Best ckpt @E{best_epoch},  val_oa {oa_when_best:.2f}, val_macc {macc_when_best:.2f}, val_miou {best_val:.2f}, '
            f'\niou per cls is: {ious_when_best}')


def train_student(gpu, cfg):
    teacher = build_model_from_cfg(cfg.model).to(cfg.rank)

    student = build_model_from_cfg(cfg.model).to(cfg.rank)
    model_size = cal_model_parm_nums(student)
    logging.info(student)
    logging.info('Number of params: %.4f M' % (model_size / 1e6))
    model_info(student)

    if cfg.sync_bn:
        student = torch.nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = torch.nn.SyncBatchNorm.convert_sync_batchnorm(teacher)
        logging.info('Using Synchronized BatchNorm ...')
    if cfg.distributed:
        torch.cuda.set_device(gpu)
        student = nn.parallel.DistributedDataParallel(student.cuda(), device_ids=[cfg.rank], output_device=cfg.rank,
                                                      find_unused_parameters=True)
        teacher = nn.parallel.DistributedDataParallel(teacher.cuda(), device_ids=[cfg.rank], output_device=cfg.rank,
                                                      find_unused_parameters=True)
        logging.info('Using Distributed Data parallel ...')

    _, _ = load_checkpoint(teacher, './teacher/Teacher_ckpt.pth')
    teacher.eval()

    # optimizer & scheduler
    optimizer = build_optimizer_from_cfg(student, lr=cfg.lr, **cfg.optimizer)
    scheduler = build_scheduler_from_cfg(cfg, optimizer)

    # build dataset
    val_loader = build_dataloader_from_cfg(cfg.get('val_batch_size', cfg.batch_size),
                                           cfg.dataset,
                                           cfg.dataloader,
                                           datatransforms_cfg=cfg.datatransforms,
                                           split='val',
                                           distributed=cfg.distributed
                                           )
    logging.info(f"length of validation dataset: {len(val_loader.dataset)}")

    train_loader = build_dataloader_from_cfg(cfg.batch_size,
                                             cfg.dataset,
                                             cfg.dataloader,
                                             datatransforms_cfg=cfg.datatransforms,
                                             split='train',
                                             distributed=cfg.distributed,
                                             )
    logging.info(f"length of training dataset: {len(train_loader.dataset)}")

    # ===> start training
    val_miou, val_macc, val_oa, val_ious, val_accs = 0., 0., 0., [], []
    best_val, macc_when_best, oa_when_best, ious_when_best, best_epoch = 0., 0., 0., [], 0
    for epoch in range(cfg.start_epoch, cfg.epochs + 1):
        if cfg.distributed:
            train_loader.sampler.set_epoch(epoch)
        if hasattr(train_loader.dataset, 'epoch'):  # some dataset sets the dataset length as a fixed steps.
            train_loader.dataset.epoch = epoch - 1
        train2d_loss, train3d_loss, train_miou2d, train_miouSC, train_miouSSC, train_iousSSC = \
            train_one_epoch(student, train_loader, optimizer, scheduler, epoch, cfg, mode='S', teacher=teacher)

        is_best = False
        cfg.val_freq = 1
        if epoch % cfg.val_freq == 0:
            val_miou2d, val_ious2d, val_miouSC, val_miouSSC, val_iousSSC, val_accsSSC = validate(student, val_loader,
                                                                                                 cfg, mode='S')
            if val_miouSSC > best_val:
                is_best = True
                best_val = val_miouSSC
                macc_when_best = val_macc
                oa_when_best = val_oa
                ious_when_best = val_ious
                best_epoch = epoch
                with np.printoptions(precision=2, suppress=True):
                    logging.info(
                        f'Find a better ckpt @E{epoch}, val_miou2d {val_miou2d:.2f} \nval_ious2d: {val_ious2d}'
                        f'\nvalSC_miou {val_miouSC:.2f} valSSC_miou {val_miouSSC:.2f}'
                        f'\nval_iousSSC: {val_iousSSC}')

        lr = optimizer.param_groups[0]['lr']
        logging.info(f'Epoch {epoch} LR {lr:.6f} '
                     f'train_miou {train_miou2d:.2f}, train_miouSC {train_miouSC:.2f}, '
                     f'train_miouSSC {train_miouSSC:.2f},'
                     f'val_miou {val_miou2d:.2f}, val_miouSC {val_miouSC:.2f}, '
                     f'val_miouSSC {val_miouSSC:.2f}, best val SSC miou {best_val:.2f}')

        if cfg.sched_on_epoch:
            scheduler.step(epoch)
        if cfg.rank == 0:
            save_checkpoint(cfg, student, epoch, optimizer, scheduler,
                            additioanl_dict={'best_val': best_val},
                            is_best=is_best
                            )

    # validate
    with np.printoptions(precision=2, suppress=True):
        logging.info(
            f'Best ckpt @E{best_epoch},  val_oa {oa_when_best:.2f}, val_macc {macc_when_best:.2f}, val_miou {best_val:.2f}, '
            f'\niou per cls is: {ious_when_best}')
