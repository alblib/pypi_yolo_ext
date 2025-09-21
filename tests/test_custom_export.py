import unittest, os
from src.yolo_ext import yolo_model_custom_export_onnx
from ultralytics import YOLO
import onnxruntime as ort
from PIL import Image
import numpy as np


class TestYOLOModelCustomExportONNX(unittest.TestCase):
    def test_result(self):
        self.model = YOLO('yolo11n.pt')
        yolo_model_custom_export_onnx(
            self.model,
            'test.onnx'
        )

    def test_result2(self):
        self.model = YOLO('yolo11n.pt')
        yolo_model_custom_export_onnx(
            self.model,
            'test.onnx',
            quantize=True
        )

    def test_simplest(self):
        self.model = YOLO('yolo11s.pt')
        yolo_model_custom_export_onnx(
            self.model,
            'simplest.onnx',
            imgsz=320,
            #quantize=True,
            truncate_for_classes=['person'],
            #dynamic=True
        )

        # Test prediction and compare
        self.exported_model = ort.InferenceSession(
            'simplest.onnx'
        )
        img = Image.open('data/' + os.listdir('data')[0], mode='r').convert("RGB").resize((320,320))
        img = np.array(img, dtype=np.float32)
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        img = np.expand_dims(img, axis=0)  # CHW -> NCHW
        img /= 255.0
        pred = self.exported_model.run(
            ['output0'],
            {'images': img}
        )[0]
        idx = np.argsort(-pred[:, 4, :], axis=1)
        idx = idx[:, np.newaxis, :]
        pred = np.take_along_axis(pred, idx, axis=2)
        print(pred)
        # predictions = self.exported_model.predict(
        #     [os.path.join('data', name) for name in os.listdir('data')])
        # for pred in predictions:
        #     pred.show()
