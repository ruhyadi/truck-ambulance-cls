"""
Convert a PyTorch model to ONNX format.
usage:
python tools/torch_to_onnx.py \
    --ckpt_path /path/to/ckpt.pth \
    --backbone mobilenetv3 \
    --categories ambulance \
    --output_path /path/to/output.onnx
"""

import rootutils

ROOT = rootutils.autosetup()

import argparse
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import onnx
import onnxruntime as ort
import torch

from src.models.ambulance_model import AmbulanceLightningModel
from src.utils.logger import get_logger

log = get_logger()


def main(
    ckpt_path: str,
    backbone: str = "mobilenetv3",
    categories: List[str] = ["ambulance"],
    output_path: str = None,
) -> None:
    """Convert a PyTorch model to ONNX format."""
    if output_path is None:
        output_path = f"tmp/models/ambulance_classifier_{backbone}_{datetime.now().strftime('%Y%m%d%H%M%S')}.onnx"
    output_path: Path = Path(output_path)

    # create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = AmbulanceLightningModel(backbone=backbone, categories=categories)
    model = AmbulanceLightningModel.load_from_checkpoint(ckpt_path)
    model.eval()
    model.to("cpu")

    # create dummy input
    x = torch.rand(1, 3, 224, 224, requires_grad=True)

    # export the model
    torch.onnx.export(
        model,
        x,
        output_path,
        export_params=True,
        input_names=["input"],
        output_names=["output"],
        opset_version=11,
        dynamic_axes={
            "input": {0: "batch_size"},
        },
    )

    log.info(f"ONNX model saved to {output_path}")

    # check if the model is valid
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    # test the model
    status = test_model(onnx_path=str(output_path))
    if status:
        log.info("ONNX model is valid.")
    else:
        raise ValueError("ONNX model is invalid.")


def test_model(onnx_path: str) -> bool:
    """Test onnx model with dummy input."""
    log.info("Testing ONNX model...")
    dummy_input = np.random.rand(1, 3, 224, 224).astype(np.float32)
    ort_session = ort.InferenceSession(onnx_path)
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input}
    ort_outs = ort_session.run(None, ort_inputs)
    outputs = [softmax(out) for out in ort_outs]
    log.info(f"Output: {outputs}")

    return True


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a PyTorch model to ONNX format."
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to the PyTorch model checkpoint.",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="mobilenetv3",
        help="Backbone model to use.",
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        default=["ambulance"],
        help="List of categories.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to save the ONNX model.",
    )
    args = parser.parse_args()

    main(args.ckpt_path, args.backbone, args.categories, args.output_path)
