## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00046428


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0041760, -0.0041467, -0.0041760, -0.0041467, -0.0000254, 0.0000254)
1: (-0.0090471, -0.0079501, -0.0090471, -0.0079501, -0.0009514, 0.0009514)
2: (0.9656065, 0.9669229, 0.9656065, 0.9669229, -0.0011417, 0.0011417)
3: (-0.0073745, 0.0023353, -0.0073745, 0.0023353, -0.0084212, 0.0084212)
4: (-0.0008706, -0.0001322, -0.0008706, -0.0001322, -0.0006405, 0.0006405)
5: (0.0163904, 0.0171368, 0.0163904, 0.0171368, -0.0006473, 0.0006473)
6: (0.0035715, 0.0039345, 0.0035715, 0.0039345, -0.0003149, 0.0003149)
7: (-0.0083835, -0.0058671, -0.0083835, -0.0058671, -0.0021824, 0.0021824)
8: (0.0100781, 0.0120745, 0.0100781, 0.0120745, -0.0017314, 0.0017314)
9: (0.0158510, 0.0194417, 0.0158510, 0.0194417, -0.0031142, 0.0031142)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.13 + 1.41 = 2.54 seconds
status: Status.ADV_EXAMPLE
