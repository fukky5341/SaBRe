## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.01546812


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0058446, 0.0039225, -0.0058446, 0.0039225, -0.0097670, 0.0097670)
1: (0.9886119, 1.0086204, 0.9886119, 1.0086204, -0.0200085, 0.0200085)
2: (-0.0145275, 0.0049149, -0.0145275, 0.0049149, -0.0190369, 0.0190369)
3: (0.0000782, 0.0060672, 0.0000782, 0.0060672, -0.0059890, 0.0059890)
4: (-0.0058385, 0.0098988, -0.0058385, 0.0098988, -0.0157372, 0.0157372)
5: (-0.0018822, 0.0113203, -0.0018822, 0.0113203, -0.0132025, 0.0132025)
6: (-0.0073843, 0.0041070, -0.0073843, 0.0041070, -0.0114913, 0.0114913)
7: (-0.0117634, -0.0005140, -0.0117634, -0.0005140, -0.0112494, 0.0112494)
8: (-0.0126708, 0.0173390, -0.0126708, 0.0173390, -0.0298615, 0.0298615)
9: (-0.0101442, 0.0070963, -0.0101442, 0.0070963, -0.0172405, 0.0172405)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.61 + 3.82 = 5.43 seconds
status: Status.VERIFIED
relational distance
Output dim: 1, lower bound: -0.0149284, upper bound: 0.0149284
