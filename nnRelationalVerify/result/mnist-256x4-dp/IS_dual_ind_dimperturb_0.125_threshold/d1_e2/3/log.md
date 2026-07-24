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
0: (-0.0001600, 0.0011190, -0.0001600, 0.0011190, -0.0012790, 0.0012790)
1: (-0.0031650, -0.0026492, -0.0031650, -0.0026492, -0.0005159, 0.0005159)
2: (0.0327760, 0.0336070, 0.0327760, 0.0336070, -0.0008309, 0.0008309)
3: (-0.0029340, -0.0019760, -0.0029340, -0.0019760, -0.0009580, 0.0009580)
4: (-0.0021201, -0.0014114, -0.0021201, -0.0014114, -0.0006092, 0.0006092)
5: (0.0123737, 0.0135654, 0.0123737, 0.0135654, -0.0011916, 0.0011916)
6: (-0.0032562, -0.0024724, -0.0032562, -0.0024724, -0.0006400, 0.0006400)
7: (0.9758825, 0.9763688, 0.9758825, 0.9763688, -0.0004863, 0.0004863)
8: (-0.0136318, -0.0103076, -0.0136318, -0.0103076, -0.0033241, 0.0033241)
9: (0.0019635, 0.0038877, 0.0019635, 0.0038877, -0.0019241, 0.0019241)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.29 + 1.20 = 2.50 seconds
status: Status.VERIFIED
relational distance
Output dim: 7, lower bound: -0.0001971, upper bound: 0.0001971
