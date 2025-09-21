import shutil

from ultralytics import YOLO
from typing import Union, Iterable, Tuple, Optional
import os
from .static_qdq_quantization import imagenet_static_qdq_quantization_onnx
from .truncate_classes import yolo_model_truncate_classes


def yolo_model_custom_export_onnx(
        yolo_model: YOLO,
        dest: Union[str, os.PathLike[str]],
        *,
        imgsz: Union[int, Tuple[int, int]] = 640,
        dynamic: bool = False,
        simplify: bool = True,
        opset: int = 12,
        batch: int = 1,
        quantize: bool = False,
        quantize_limit: int = 256,
        truncate_for_classes: Optional[Union[Union[int, str], Iterable[Union[int, str]]]] = None,
):
    if truncate_for_classes is not None:
        yolo_model = yolo_model_truncate_classes(yolo_model, truncate_for_classes)
    yolo_mid_onnx = yolo_model.export(
        format='onnx',
        imgsz=imgsz,
        dynamic=dynamic,
        simplify=simplify,
        opset=opset,
        batch=batch
    )
    if quantize:
        imagenet_static_qdq_quantization_onnx(
            yolo_mid_onnx,
            dest,
            imgsz,
            quantize_limit
        )
    else:
        shutil.move(yolo_mid_onnx, dest)
    return dest
