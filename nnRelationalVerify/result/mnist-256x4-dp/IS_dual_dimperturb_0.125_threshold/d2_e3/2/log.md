## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0002144


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.9912901, 0.9922794, 0.9912901, 0.9922794, -0.0004820, 0.0004820)
1: (-0.0034342, -0.0031877, -0.0034342, -0.0031877, -0.0001201, 0.0001201)
2: (0.0068393, 0.0081457, 0.0068393, 0.0081457, -0.0006364, 0.0006364)
3: (-0.0049807, -0.0043861, -0.0049807, -0.0043861, -0.0002897, 0.0002897)
4: (0.0018516, 0.0021045, 0.0018516, 0.0021045, -0.0001232, 0.0001232)
5: (0.0075615, 0.0092045, 0.0075615, 0.0092045, -0.0008005, 0.0008005)
6: (-0.0007954, -0.0003784, -0.0007954, -0.0003784, -0.0002032, 0.0002032)
7: (-0.0051955, -0.0041166, -0.0051955, -0.0041166, -0.0005257, 0.0005257)
8: (-0.0022964, -0.0017290, -0.0022964, -0.0017290, -0.0002764, 0.0002764)
9: (0.0001410, 0.0007990, 0.0001410, 0.0007990, -0.0003205, 0.0003205)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.41 + 1.23 = 2.63 seconds
status: Status.ADV_EXAMPLE
