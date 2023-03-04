import __init__
import argparse, yaml, os, logging, numpy as np
import torch
from torch import distributed as dist, multiprocessing as mp
from cleaner.utils import set_random_seed, setup_logger_dist, \
    generate_exp_directory, resume_exp_directory, EasyConfig, dist_utils, find_free_port
import warnings
from train_utils import train_teacher, train_student
warnings.simplefilter(action='ignore', category=FutureWarning)


def main(gpu, cfg):
    if cfg.distributed:
        if cfg.mp:
            cfg.rank = gpu
        dist.init_process_group(backend=cfg.dist_backend,
                                init_method=cfg.dist_url,
                                world_size=cfg.world_size,
                                rank=cfg.rank)
    # logger
    setup_logger_dist(cfg.log_path, cfg.rank, name=cfg.dataset.common.NAME)
    set_random_seed(cfg.seed + cfg.rank, deterministic=cfg.deterministic)
    torch.backends.cudnn.enabled = True
    logging.info(cfg)

    # -------------------------
    # step1: train the teacher network
    # -------------------------
    if not os.path.exists('./teacher/Teacher_ckpt.pth'):
        train_teacher(gpu, cfg)

    # -------------------------
    # step2: train the student network
    # -------------------------
    train_student(gpu, cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Scene segmentation training/testing')
    parser.add_argument('--cfg', type=str, default='../../cfgs/NYU/voxelSSC.yaml', help='config file')
    args, opts = parser.parse_known_args()
    cfg = EasyConfig()
    cfg.load(args.cfg, recursive=True)
    cfg.update(opts)    # overwrite the default arguments in yml 

    if cfg.seed is None:
        cfg.seed = np.random.randint(1, 10000)

    # init distributed env first, since logger depends on the dist info.
    cfg.rank, cfg.world_size, cfg.distributed, cfg.mp = dist_utils.get_dist_info(cfg)
    cfg.sync_bn = cfg.world_size > 1

    # init log dir
    cfg.task_name = args.cfg.split('.')[-2].split('/')[-2]
    cfg.cfg_basename = args.cfg.split('.')[-2].split('/')[-1]
    tags = [
        cfg.task_name,  # task name
        cfg.mode,
        cfg.cfg_basename,  # cfg file name
        f'ngpus{cfg.world_size}',
        f'seed{cfg.seed}',
    ]
    for i, opt in enumerate(opts):
        if 'rank' not in opt and 'dir' not in opt and 'root' not in opt and 'path' not in opt and 'wandb' not in opt and '/' not in opt:
            tags.append(opt)
    cfg.root_dir = os.path.join(cfg.root_dir, cfg.task_name)

    cfg.is_training = cfg.mode in ['train', 'training', 'finetune', 'finetuning']
    if cfg.mode == 'train':
        generate_exp_directory(cfg, tags, additional_id=os.environ.get('MASTER_PORT', None))
        cfg.wandb.tags = tags
    else:  # resume from the existing ckpt and reuse the folder.
        resume_exp_directory(cfg, pretrained_path=cfg.pretrained_path)
        cfg.wandb.tags = ['resume']
    os.environ["JOB_LOG_DIR"] = cfg.log_dir
    cfg_path = os.path.join(cfg.run_dir, "cfg.yaml")
    with open(cfg_path, 'w') as f:
        yaml.dump(cfg, f, indent=2)
        os.system('cp %s %s' % (args.cfg, cfg.run_dir))
    cfg.cfg_path = cfg_path

    # multi processing.
    if cfg.mp:
        port = find_free_port()
        cfg.dist_url = f"tcp://localhost:{port}"
        print('using mp spawn for distributed training')
        mp.spawn(main, nprocs=cfg.world_size, args=(cfg, ))
    else:
        main(0, cfg)
