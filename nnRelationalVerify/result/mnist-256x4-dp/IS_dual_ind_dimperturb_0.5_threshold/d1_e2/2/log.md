## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00146637


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0040746, -0.0037927, -0.0040746, -0.0037927, -0.0002819, 0.0002819)
1: (-0.0052493, -0.0043796, -0.0052493, -0.0043796, -0.0008697, 0.0008697)
2: (0.9697933, 0.9710031, 0.9697933, 0.9710031, -0.0012097, 0.0012097)
3: (0.0262408, 0.0324292, 0.0262408, 0.0324292, -0.0045721, 0.0045721)
4: (-0.0031595, -0.0026170, -0.0031595, -0.0026170, -0.0005425, 0.0005425)
5: (0.0138652, 0.0145528, 0.0138652, 0.0145528, -0.0006877, 0.0006877)
6: (0.0045919, 0.0050597, 0.0045919, 0.0050597, -0.0004678, 0.0004678)
7: (-0.0161826, -0.0145788, -0.0161826, -0.0145788, -0.0016038, 0.0016038)
8: (0.0038907, 0.0051630, 0.0038907, 0.0051630, -0.0012724, 0.0012724)
9: (0.0045354, 0.0070108, 0.0045354, 0.0070108, -0.0024754, 0.0024754)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.22 + 1.45 = 2.67 seconds
status: Status.VERIFIED
relational distance
Output dim: 2, lower bound: -0.0010886, upper bound: 0.0010886
