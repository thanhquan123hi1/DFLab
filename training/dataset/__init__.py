import importlib
import os
import sys

current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
project_root_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(project_root_dir)

from .abstract_dataset import DeepfakeAbstractBaseDataset


def _lazy_dataset(module_name, class_name):
    class _LazyDataset:
        def __new__(cls, *args, **kwargs):
            module = importlib.import_module(f"{__name__}.{module_name}")
            dataset_class = getattr(module, class_name)
            return dataset_class(*args, **kwargs)

    _LazyDataset.__name__ = class_name
    return _LazyDataset


I2GDataset = _lazy_dataset("I2G_dataset", "I2GDataset")
IIDDataset = _lazy_dataset("iid_dataset", "IIDDataset")
FFBlendDataset = _lazy_dataset("ff_blend", "FFBlendDataset")
FWABlendDataset = _lazy_dataset("fwa_blend", "FWABlendDataset")
LRLDataset = _lazy_dataset("lrl_dataset", "LRLDataset")
pairDataset = _lazy_dataset("pair_dataset", "pairDataset")
SBIDataset = _lazy_dataset("sbi_dataset", "SBIDataset")
LSDADataset = _lazy_dataset("lsda_dataset", "LSDADataset")
TALLDataset = _lazy_dataset("tall_dataset", "TALLDataset")
