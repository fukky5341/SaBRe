## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0041507


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=22, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.9911307, 0.9984838, 0.9911307, 0.9984838, -0.0073531, 0.0073531)
1: (-0.0034740, -0.0016418, -0.0034740, -0.0016418, -0.0018322, 0.0018322)
2: (-0.0013535, 0.0083561, -0.0013535, 0.0083561, -0.0097097, 0.0097097)
3: (-0.0050765, -0.0006571, -0.0050765, -0.0006571, -0.0044194, 0.0044194)
4: (0.0002659, 0.0021452, 0.0002659, 0.0021452, -0.0018793, 0.0018793)
5: (-0.0027429, 0.0094693, -0.0027429, 0.0094693, -0.0122122, 0.0122122)
6: (-0.0008626, 0.0022370, -0.0008626, 0.0022370, -0.0030996, 0.0030996)
7: (-0.0053694, 0.0026502, -0.0053694, 0.0026502, -0.0080196, 0.0080196)
8: (-0.0023878, 0.0018296, -0.0023878, 0.0018296, -0.0042174, 0.0042174)
9: (-0.0039853, 0.0009050, -0.0039853, 0.0009050, -0.0048903, 0.0048903)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.58 + 2.10 = 3.68 seconds
status: Status.VERIFIED
relational distance
Output dim: 0, lower bound: -0.0040652, upper bound: 0.0040652
