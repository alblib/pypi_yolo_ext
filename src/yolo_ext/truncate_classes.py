from torch import nn
from typing import Union, Iterable
from ultralytics import YOLO
import numpy as np


def yolo_model_truncate_classes(
        model: Union[YOLO, nn.Module],
        class_ids: Union[Union[int, str], Iterable[Union[int, str]]]
) -> Union[YOLO, nn.Module]:
    """
    Shrink a YOLO detection model with reduced classes.
    :param model: An input YOLO detection model.
    :param class_ids: A class or a list of class to be chosen from the names of the given model.
                      This can be class ids or names.
    :return: A new YOLO model with only specified class outputs.
    """

    # Normalize parameters
    if isinstance(class_ids, int) or isinstance(class_ids, str):
        class_ids = [class_ids]
    else:
        class_ids = list(class_ids)
    reverse_name_dict = {v: k for k, v in model.model.names.items()}
    for i in range(len(class_ids)):
        if isinstance(class_ids[i], str):
            class_ids[i] = reverse_name_dict[class_ids[i]]
    class_ids = np.array(class_ids)

    # last layer slice
    last_layer = model.model.model[-1]
    cv3 = last_layer.cv3
    #dfl = last_layer.dfl
    for cv3_mod_idx in range(len(cv3)):
        module = cv3[cv3_mod_idx][-1]
        module.out_channels = len(class_ids)
        module.weight = nn.Parameter(module.weight[class_ids])
        module.bias = nn.Parameter(module.bias[class_ids])
        cv3[cv3_mod_idx][-1] = module
    last_layer.cv3 = cv3
    last_layer.nc = len(class_ids)
    last_layer.no = 64 + len(class_ids)
    model.model.model[-1] = last_layer
    model.model.nc = len(class_ids)
    model.model.names = {
        order: model.model.names[class_id]
        for order, class_id in enumerate(class_ids)
    }

    return model
