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
Threshold: 0.00527912


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0008362, 0.0015766, -0.0008362, 0.0015766, -0.0024128, 0.0024128)
1: (-0.0034378, -0.0024647, -0.0034378, -0.0024647, -0.0009731, 0.0009731)
2: (0.0323367, 0.0339042, 0.0323367, 0.0339042, -0.0015675, 0.0015675)
3: (-0.0032768, -0.0014695, -0.0032768, -0.0014695, -0.0018073, 0.0018073)
4: (-0.0023736, -0.0010367, -0.0023736, -0.0010367, -0.0011632, 0.0011632)
5: (0.0117437, 0.0139917, 0.0117437, 0.0139917, -0.0022479, 0.0022479)
6: (-0.0036706, -0.0021921, -0.0036706, -0.0021921, -0.0012283, 0.0012283)
7: (0.9757085, 0.9766259, 0.9757085, 0.9766259, -0.0009174, 0.0009174)
8: (-0.0148208, -0.0085501, -0.0148208, -0.0085501, -0.0062707, 0.0062707)
9: (0.0009462, 0.0045760, 0.0009462, 0.0045760, -0.0036297, 0.0036297)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.18 + 1.36 = 2.54 seconds
status: Status.VERIFIED
relational distance
Output dim: 7, lower bound: -0.0004666, upper bound: 0.0004667
