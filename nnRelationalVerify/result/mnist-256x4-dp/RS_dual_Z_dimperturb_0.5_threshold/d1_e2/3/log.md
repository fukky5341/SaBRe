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
Threshold: 0.00077259


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0032337, 0.0023911, -0.0032337, 0.0023911, -0.0056247, 0.0056247)
1: (-0.0043901, -0.0021362, -0.0043901, -0.0021362, -0.0022539, 0.0022539)
2: (0.0309216, 0.0344334, 0.0309216, 0.0344334, -0.0035117, 0.0035117)
3: (-0.0038868, 0.0007935, -0.0038868, 0.0007935, -0.0038539, 0.0038539)
4: (-0.0028248, 0.0002536, -0.0028248, 0.0002536, -0.0030785, 0.0030785)
5: (0.0097143, 0.0147505, 0.0097143, 0.0147505, -0.0050361, 0.0050361)
6: (-0.0054020, -0.0016930, -0.0054020, -0.0016930, -0.0037090, 0.0037090)
7: (0.9748227, 0.9774543, 0.9748227, 0.9774543, -0.0026316, 0.0026316)
8: (-0.0169376, -0.0028892, -0.0169376, -0.0028892, -0.0140484, 0.0140484)
9: (-0.0023306, 0.0058013, -0.0023306, 0.0058013, -0.0081319, 0.0081319)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.13 + 1.29 = 2.42 seconds
status: Status.ADV_EXAMPLE
