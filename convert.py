import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torchvision.models as models

# Download pretrained ResNet18 (trained on ImageNet - 1000 object classes)
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.eval()  # switch to inference mode, not training mode

# Dummy input - simulates a 224x224 RGB image
dummy_input = torch.randn(1, 3, 224, 224)

# Convert and save as ONNX
torch.onnx.export(
    model,
    dummy_input,
    "/Users/I760158/ml_inference_env/resnet18.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=12
)

print("Model converted and saved as resnet18.onnx")
