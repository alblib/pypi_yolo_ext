import unittest
from src.yolo_ext import imagenet_static_qdq_quantization_onnx
from ultralytics import YOLO


class TestYOLOModelStaticQDQQuantization(unittest.TestCase):
    def test_result(self):
        self.model = YOLO('yolo11n.pt')
        self.model.export(format='onnx')
        imagenet_static_qdq_quantization_onnx(
            'yolo11n.onnx',
            'yolo11n_q8.onnx',
            640,
            256
        )
