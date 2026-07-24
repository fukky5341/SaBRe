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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0009536, 0.0014532, -0.0009536, 0.0014532, -0.0024068, 0.0024068)
1: (-0.0034851, -0.0025144, -0.0034851, -0.0025144, -0.0009707, 0.0009707)
2: (0.0322605, 0.0338241, 0.0322605, 0.0338241, -0.0015636, 0.0015636)
3: (-0.0031843, -0.0013816, -0.0031843, -0.0013816, -0.0018027, 0.0018027)
4: (-0.0023052, -0.0009717, -0.0023052, -0.0009717, -0.0011662, 0.0011662)
5: (0.0116344, 0.0138767, 0.0116344, 0.0138767, -0.0022423, 0.0022423)
6: (-0.0037425, -0.0022677, -0.0037425, -0.0022677, -0.0012334, 0.0012334)
7: (0.9757554, 0.9766707, 0.9757554, 0.9766707, -0.0009153, 0.0009153)
8: (-0.0145002, -0.0082452, -0.0145002, -0.0082452, -0.0062551, 0.0062551)
9: (0.0007697, 0.0043904, 0.0007697, 0.0043904, -0.0036207, 0.0036207)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.17 + 1.31 = 2.48 seconds
status: Status.VERIFIED
relational distance
Output dim: 7, lower bound: -0.0004636, upper bound: 0.0004637
