## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 1)
Time budget: 60 seconds
Split limit: 100
Threshold: 0.006374325790271126


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0041227, -0.0027547, -0.0041227, -0.0027547, -0.0013680, 0.0013680)
1: (-0.0070515, -0.0033202, -0.0070515, -0.0033202, -0.0037313, 0.0037313)
2: (0.9655445, 0.9714753, 0.9655445, 0.9714753, -0.0059308, 0.0059308)
3: (0.0102895, 0.0359129, 0.0102895, 0.0359129, -0.0183497, 0.0183497)
4: (-0.0034244, -0.0007644, -0.0034244, -0.0007644, -0.0026600, 0.0026600)
5: (0.0127701, 0.0157790, 0.0127701, 0.0157790, -0.0030089, 0.0030089)
6: (0.0028963, 0.0051899, 0.0028963, 0.0051899, -0.0022936, 0.0022936)
7: (-0.0170854, -0.0104448, -0.0170854, -0.0104448, -0.0066405, 0.0066405)
8: (0.0031744, 0.0084427, 0.0031744, 0.0084427, -0.0052683, 0.0052683)
9: (0.0025176, 0.0129096, 0.0025176, 0.0129096, -0.0103920, 0.0103920)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.09 + 1.22 = 2.31 seconds
status: Status.VERIFIED
relational distance
Output dim: 2, lower bound: -0.0053060, upper bound: 0.0053060
