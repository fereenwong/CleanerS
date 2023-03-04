import os
import logging
import numpy as np
from torch.utils.data import Dataset
from ..build import DATASETS
import cv2
import imageio


@DATASETS.register_module()
class NYU(Dataset):
    classes = ['empty', 'ceiling', 'floor', 'wall', 'window', 'chair', 'bed', 'sofa', 'table', 'tvs', 'furn', 'objs']
    num_classes = 12
    class2color = {'empty': [22, 191, 206],
                   'ceiling': [214, 38, 40],
                   'floor': [43, 160, 4],
                   'wall': [158, 216, 229],
                   'window': [114, 158, 206],
                   'chair': [204, 204, 91],
                   'bed': [255, 186, 119],
                   'sofa': [147, 102, 188],
                   'table': [30, 119, 181],
                   'tvs': [188, 188, 33],
                   'furn': [255, 127, 12],
                   'objects': [196, 175, 214],
                   'ignore': [153, 153, 153]}
    cmap = [*class2color.values()]

    def __init__(self,
                 data_root: str = '/path-to-data/SSC/NYU',
                 img_H: int = 480,
                 img_W: int = 640,
                 split: str = 'train',
                 transform=None,
                 ):

        super().__init__()
        self.split, self.transform, self.data_root = \
            split, transform, data_root
        self.img_H = img_H
        self.img_W = img_W

        data_list = os.listdir(os.path.join(data_root, split, 'RGB'))
        self.data_list = [item[:7] for item in data_list if '.png' in item]

        self.data_idx = np.arange(len(self.data_list))
        assert len(self.data_idx) > 0
        logging.info(f"\nTotally {len(self.data_idx)} samples in {split} set")

    def load_img(self, item):
        rgb_path = os.path.join(self.data_root, self.split, 'RGB', '{}_rgb.png'.format(item))
        img = np.array(cv2.imread(rgb_path), dtype=np.float32)
        return img

    def load_mapping(self, item):
        mapping_path = os.path.join(self.data_root, 'Mapping', '{}.npz'.format(item[3:]))
        mapping = np.load(mapping_path)['arr_0'].astype(np.int32)

        mapping2d = (np.ones((self.img_H, self.img_W)) * -1).reshape(-1).astype(np.long)
        mapping2d[mapping[mapping != 307200]] = np.nonzero(mapping != 307200)[0]
        mapping2d = mapping2d.reshape(self.img_H, self.img_W)
        return mapping, mapping2d

    def load_tsdf(self, item, CAD=False):
        tsdf_path = os.path.join(self.data_root, 'TSDF', '{}.npz'.format(item[3:]))

        if CAD:
            tsdf_path = tsdf_path.replace('NYU', 'NYUCAD')

        tsdf = np.load(tsdf_path)['arr_0'].astype(np.float32).reshape(1, 60, 36, 60)
        return tsdf

    def load_label(self, item):
        label3d_path = os.path.join(self.data_root, 'Label', '{}.npz'.format(item[3:]))
        labelweight_path = os.path.join(self.data_root, 'TSDF', '{}.npz'.format(item[3:]))

        label3d = np.load(label3d_path)['arr_0'].astype(np.long)
        label_weight = np.load(labelweight_path)['arr_1'].astype(np.float32)
        return label3d, label_weight

    def load_depth(self, item, CAD=False):
        data_root = self.data_root.replace('NYU', 'NYUCAD') if CAD else self.data_root
        depth_path = os.path.join(data_root, self.split, 'depth', '{}_0000.png'.format(item))
        depth = imageio.imread(depth_path) / 8000.0
        depth = np.array(depth)
        return depth

    def __getitem__(self, idx):
        data_idx = self.data_idx[idx % len(self.data_idx)]

        item = self.data_list[data_idx]
        img = self.load_img(item)
        label3d, label_weight = self.load_label(item)

        # Notes: the same mapping for both teacher and student models
        mapping, mapping2d = self.load_mapping(item)
        tsdf, tsdf_CAD = self.load_tsdf(item, CAD=False), self.load_tsdf(item, CAD=True)

        data = {'img': img,
                'label3d': label3d, 'label_weight': label_weight,
                'mapping': mapping, 'mapping2d': mapping2d,
                'file': item, 'tsdf': tsdf, 'tsdf_CAD': tsdf_CAD}
        # pre-process.
        if self.transform is not None:
            data = self.transform(data)
        return data

    def __len__(self):
        return len(self.data_idx)
