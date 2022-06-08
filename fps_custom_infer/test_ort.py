import onnxruntime
import numpy as np

device_name = 'cuda:0' # or 'cpu'

if device_name == 'cpu':
    providers = ['CPUExecutionProvider']
elif device_name == 'cuda:0':
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
# Create inference session
onnx_model = onnxruntime.InferenceSession('mydomain::fps_model1.onnx', providers=providers)
# Create the input（这里的输入对应slowfast的输入）
data = np.random.rand(1,16384,4).astype(np.float32)
# Inference
onnx_input = {onnx_model.get_inputs()[0].name: data}
outputs = onnx_model.run(None, onnx_input)

