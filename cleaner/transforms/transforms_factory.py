from ..utils.registry import Registry
import torchvision.transforms as transforms

DataTransforms = Registry('datatransforms')


def build_transforms_from_cfg(split, datatransforms_cfg):
    """
    Build a dataset transform for a certrain split, defined by `datatransforms_cfg`.
    """
    transform_list = datatransforms_cfg.get(split, None)
    transform_args = datatransforms_cfg.get('kwargs', None)
    if transform_list is None or len(transform_list) == 0:
        return None
    data_transforms = []
    if len(transform_list) > 1:
        for t in transform_list:
            data_transforms.append(DataTransforms.build(
                {'NAME': t}, default_args=transform_args))
        return transforms.Compose(data_transforms)
    else:
        return DataTransforms.build({'NAME': transform_list[0]}, default_args=transform_args)
