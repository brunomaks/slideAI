# Contributors:
# - Pavlo

import tensorflow as tf
import torch
import tf2onnx
import onnx 
from onnx2pytorch import ConvertModel 

MODEL_PATH = "../../shared_artifacts/models/gesture_model_20251221_184630.keras"

# Load tensorflow model 
tf_model = tf.keras.models.load_model(MODEL_PATH)
tf_model.output_names=['output']


# Convert model to onnx format
onnx_model, _ = tf2onnx.convert.from_keras(tf_model)

# Convert ONNX model to PyTorch 
pytorch_model = ConvertModel(onnx_model) 

torch.save(pytorch_model, "pytorch_model_full.pth")
