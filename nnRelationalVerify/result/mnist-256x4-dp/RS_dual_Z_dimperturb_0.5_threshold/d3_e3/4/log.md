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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=22, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.9911419, 0.9982943, 0.9911419, 0.9982943, -0.0071228, 0.0071228)
1: (-0.0034712, -0.0016890, -0.0034712, -0.0016890, -0.0017748, 0.0017748)
2: (-0.0011033, 0.0083415, -0.0011033, 0.0083415, -0.0094056, 0.0094056)
3: (-0.0050698, -0.0007710, -0.0050698, -0.0007710, -0.0042810, 0.0042810)
4: (0.0003144, 0.0021424, 0.0003144, 0.0021424, -0.0018204, 0.0018204)
5: (-0.0024282, 0.0094508, -0.0024282, 0.0094508, -0.0118298, 0.0118298)
6: (-0.0008579, 0.0021571, -0.0008579, 0.0021571, -0.0030025, 0.0030025)
7: (-0.0053572, 0.0024435, -0.0053572, 0.0024435, -0.0077684, 0.0077684)
8: (-0.0023815, 0.0017209, -0.0023815, 0.0017209, -0.0040853, 0.0040853)
9: (-0.0038593, 0.0008976, -0.0038593, 0.0008976, -0.0047372, 0.0047372)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.84 + 2.08 = 3.92 seconds
status: Status.VERIFIED
relational distance
Output dim: 0, lower bound: -0.0041447, upper bound: 0.0041447
