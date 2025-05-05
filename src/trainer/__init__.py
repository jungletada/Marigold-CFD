# Author: Bingxin Ke
# Last modified: 2024-05-17

from .marigold_trainer import MarigoldTrainer
from .cfd_trainer import CFDTrainer

trainer_cls_name_dict = {
    "MarigoldTrainer": MarigoldTrainer,
    "CFDTrainer": CFDTrainer,
}


def get_trainer_cls(trainer_name):
    return trainer_cls_name_dict[trainer_name]
