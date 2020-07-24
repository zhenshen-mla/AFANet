from thop import profile
from thop import clever_format
import torch
import make_network


def flops_params(model, input):
    flops, params = profile(model, inputs=(input,))
    flop, param = clever_format([flops, params], "%.3f")
    print(param, flop)


if __name__ == '__main__':
    model = make_network()
    input = torch.randn(1, 3, 224, 224)
    flops_params(model, input)
