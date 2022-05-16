import torch, os
import torch.nn as nn
from torch.autograd import Function, Variable
import torch
import fps
from torch.onnx import register_custom_op_symbolic
print(torch.cuda.device_count())

# def register_custom_op():
#     def my_fps(g, xyz: torch.Tensor, npoint: int) -> torch.Tensor:
#         assert xyz.is_contiguous()
#         B, N, _ = xyz.size()
#         output = torch.cuda.IntTensor(B, npoint)
#         temp = torch.cuda.FloatTensor(B, N).fill_(1e10)
#         fps.furthest_point_sampling_wrapper(B, N, npoint, xyz, temp, output)
#         return g.op("mydomain::fps", xyz, npoint_i=npoint)
#     register_custom_op_symbolic("mynamespace::furthest_point_sampling_wrapper", my_fps, 9)

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
class FurthestPointSampling(Function):
    @staticmethod
    def forward(ctx, xyz: torch.Tensor, npoint: int) -> torch.Tensor:
        """
        Uses iterative furthest point sampling to select a set of npoint features that have the largest
        minimum distance
        :param ctx:
        :param xyz: (B, N, 3) where N > npoint
        :param npoint: int, number of features in the sampled set
        :return:
             output: (B, npoint) indices of the sampled points
        """
        assert xyz.is_contiguous()

        B, N, _ = xyz.size()
        output = torch.cuda.IntTensor(B, npoint)
        temp = torch.cuda.FloatTensor(B, N).fill_(1e10)

        fps.furthest_point_sampling_wrapper(B, N, npoint, xyz, temp, output)
        return output

    @staticmethod
    def backward(xyz, a=None):
        # no backward gradient flow through FPS
        return None, None

    @staticmethod
    def symbolic(g, xyz, npoint):
        """
        Note:
            the parameters in symbolic need to be the same as those in forward
        """
        return g.op("mydomain::fps", xyz, npoint_i=npoint)

furthest_point_sample = FurthestPointSampling.apply


def export_custom_op():
    class CustomModel(torch.nn.Module):
        def __init__(self, x2):
            super().__init__()
            self.x2 = x2
        def forward(self, x1):
            output = furthest_point_sample(x1, self.x2)
            print(output)
            return output

    x1 = torch.randn(1, 6, 3).cuda()
    print(x1)
    x2 = 3
    model = CustomModel(x2)

    f = './model.onnx'
    torch.onnx.export(model, (x1,), f,
                      opset_version=12,
                      example_outputs=None,
                      input_names=["X1"], output_names=["Y"])

register_custom_op()
export_custom_op()