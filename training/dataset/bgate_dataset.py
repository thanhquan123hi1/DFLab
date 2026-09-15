"""Baseline transforms with Albumentations 2 RNG controlled per augmented frame."""
import random
from dataset.abstract_dataset import DeepfakeAbstractBaseDataset


def seed_transform(transform):
    # Albumentations 1 uses global RNGs; v2 owns independent generators.
    if hasattr(transform, 'set_random_seed'):
        transform.set_random_seed(random.getrandbits(32))


class BGateDataset(DeepfakeAbstractBaseDataset):
    def data_aug(self, img, landmark=None, mask=None, augmentation_seed=None):
        seed_transform(self.transform)
        return super().data_aug(img, landmark, mask, augmentation_seed)
