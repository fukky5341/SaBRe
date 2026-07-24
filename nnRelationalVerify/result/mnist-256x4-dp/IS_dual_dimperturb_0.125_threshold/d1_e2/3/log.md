## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.000228159


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0002410, 0.0010190, -0.0002410, 0.0010190, -0.0012600, 0.0012600)
1: (-0.0031977, -0.0026895, -0.0031977, -0.0026895, -0.0005082, 0.0005082)
2: (0.0327234, 0.0335420, 0.0327234, 0.0335420, -0.0008186, 0.0008186)
3: (-0.0028591, -0.0019154, -0.0028591, -0.0019154, -0.0009438, 0.0009438)
4: (-0.0020647, -0.0013666, -0.0020647, -0.0013666, -0.0006045, 0.0006045)
5: (0.0122983, 0.0134722, 0.0122983, 0.0134722, -0.0011739, 0.0011739)
6: (-0.0033058, -0.0025337, -0.0033058, -0.0025337, -0.0006368, 0.0006368)
7: (0.9759205, 0.9763996, 0.9759205, 0.9763996, -0.0004791, 0.0004791)
8: (-0.0133718, -0.0100971, -0.0133718, -0.0100971, -0.0032746, 0.0032746)
9: (0.0018417, 0.0037372, 0.0018417, 0.0037372, -0.0018955, 0.0018955)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.26 + 1.20 = 2.46 seconds
status: Status.VERIFIED
relational distance
Output dim: 7, lower bound: -0.0002069, upper bound: 0.0002069
