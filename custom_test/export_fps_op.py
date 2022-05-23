# SPDX-License-Identifier: Apache-2.0

import torch
from torch.onnx import register_custom_op_symbolic
# from yaml import parse
from torch.onnx.symbolic_helper import parse_args

print(torch.cuda.is_available())
print(torch.__version__)
print(torch.version.cuda)

torch.ops.load_library("/root/workspace/onnxruntime_inference_test/custom_test/build/lib.linux-x86_64-3.8/fps.cpython-38-x86_64-linux-gnu.so")

def register_custom_op():
    def fps(g, B, N, npoint, xyz, temp, output):
        return g.op("mydomain::fps", B, N, npoint, xyz, temp, output)
    register_custom_op_symbolic("mynamespace::furthest_point_sampling_wrapper", fps, 9)

def export_custom_op():
    class CustomModel(torch.nn.Module):
        def forward(self, B,N,npoint,x1,temp,output):
            return torch.ops.mynamespace.furthest_point_sampling_wrapper(B, N, npoint, x1,temp,output)
    x1 = torch.randn(1, 6, 3).cuda()
    print(x1)
    npoint = 3
    B, N, _ = x1.size()
    output = torch.cuda.IntTensor(B, npoint).long()
    temp = torch.cuda.FloatTensor(B, N).fill_(1e10)
    f = './model.onnx'
    torch.onnx.export(CustomModel(), (B,N,npoint,x1,temp,output), f,
                      opset_version=12,
                      verbose=True, 
                      input_names=["x1", "temp", "output"], output_names=["Y"],
                      custom_opsets={"mydomain": 1})

register_custom_op()
export_custom_op()

