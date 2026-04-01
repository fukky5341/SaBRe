import torch
from common.network import LayerType


class DummyLayer:
    def __init__(self, layer_type, weight=None, bias=None,
                 stride=(1, 1), padding=(0, 0), dilation=(1, 1)):
        self.type = layer_type
        self.weight = weight      # torch.Tensor | None
        self.bias = bias        # torch.Tensor | None
        self.stride = stride
        self.padding = padding
        self.dilation = dilation


def get_net_examle():

    w1 = torch.tensor([[1.0, -1.0], [-2.0, 1.0]])
    b1 = torch.tensor([0.0, 0.0])
    w2 = torch.tensor([[1.0, -1.0]])
    b2 = torch.tensor([0.0])

    net = [
        DummyLayer(LayerType.Linear, weight=w1, bias=b1),
        DummyLayer(LayerType.ReLU),
        DummyLayer(LayerType.Linear, weight=w2, bias=b2)
    ]

    return net