import os, shutil, tarfile, urllib.request
import random
from typing import Union, List, Tuple, Iterable


def _download(
        url: str,
        dest: Union[str, os.PathLike[str]]
) -> Union[str, os.PathLike[str]]:
    """
    Download a file from the internet.
    :param url: Online file url
    :param dest: Destination file path in local
    :return: `dest`
    """
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest) and os.stat(dest).st_size > 0: return dest
    print(f"Downloading: {url}")
    tmp = os.path.splitext(dest)[0] + '.part'
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    os.replace(tmp, dest)
    return dest


def _extract_tgz(
        tgz_path: Union[str, os.PathLike[str]],
        out_dir: Union[str, os.PathLike[str]]
) -> Union[str, os.PathLike[str]]:
    os.makedirs(out_dir, exist_ok=True)
    with tarfile.open(tgz_path, "r:gz") as tf: tf.extractall(out_dir)
    roots = [
        p for p in os.listdir(out_dir)
        if os.path.isdir(p) and os.path.basename(p).startswith("imagenette2")
    ]
    return roots[0] if roots else out_dir

def prepare_auto_calib(
        cache_root: Union[str, os.PathLike[str]]
) -> Union[str, os.PathLike[str]]:
    url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
    cache_dir = os.path.join(cache_root, "auto_calib")
    archive = os.path.join(cache_dir, os.path.basename(url))
    _download(url, archive)
    data_root = _extract_tgz(archive, cache_dir)
    return data_root



def _pick_model_io(onnx_path: Union[str, os.PathLike[str]]) -> Tuple[str, Tuple[int,int]]:
    import onnx
    m = onnx.load(str(onnx_path))
    for vi in m.graph.input:
        t = vi.type.tensor_type
        if t.HasField("shape") and len(t.shape.dim) == 4:
            name = vi.name
            dims = t.shape.dim
            H = int(dims[2].dim_value) if dims[2].HasField("dim_value") else None
            W = int(dims[3].dim_value) if dims[3].HasField("dim_value") else None
            return name, (H or 0, W or 0)
    return m.graph.input[0].name, (0,0)

class FolderImageReader:
    """CalibrationDataReader for ORT static quantization (expects NCHW float32 [0,1])."""
    def __init__(
            self,
            input_name: str,
            folder: Union[str, os.PathLike[str]],
            imgsz: Union[int, Tuple[int, int], List[int]],
            limit: int,
            shuffle: bool=True
    ):
        self.input_name = input_name
        self.imgsz = imgsz if isinstance(imgsz, Iterable) else (imgsz, imgsz)
        exts = (".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff")
        files = []
        for dirpath, _, filenames in os.walk(folder):
            for name in filenames:
                if name.lower().endswith(exts):
                    files.append(os.path.join(dirpath, name))
        if shuffle: random.shuffle(files)
        if limit > 0: files = files[:limit]
        if not files: raise SystemExit(f"No images found under: {folder}")
        self.files = files; self._idx = 0
    def get_next(self):
        if self._idx >= len(self.files): return None
        path = self.files[self._idx]; self._idx += 1
        import numpy as np
        from PIL import Image
        img = Image.open(path).convert("RGB").resize(self.imgsz)
        arr = (np.asarray(img, dtype=np.float32) / 255.0).transpose(2,0,1)[None, ...]
        return {self.input_name: arr}



def imagenet_static_qdq_quantization_onnx(
        onnx_in: Union[str, os.PathLike[str]],
        onnx_out: Union[str, os.PathLike[str]],
        imgsz: Union[int, Tuple[int, int], List[int]],
        limit: int = 256
):
    cache_dir = os.path.join(os.path.dirname(onnx_out), 'auto_calib')
    prepare_auto_calib(cache_dir)

    input_name, _ = _pick_model_io(onnx_in)
    reader = FolderImageReader(input_name, cache_dir, imgsz=imgsz, limit=limit, shuffle=True)
    from onnxruntime.quantization import quantize_static, QuantType, CalibrationMethod, QuantFormat
    quantize_static(
        model_input=str(onnx_in),
        model_output=str(onnx_out),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        per_channel=True,
        reduce_range=False
    )
    return onnx_out
