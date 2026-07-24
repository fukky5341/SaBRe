## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.13486788544


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0573947, 0.0236576, -0.0573947, 0.0236576, -0.0810523, 0.0810523)
1: (-0.0525208, 0.0371929, -0.0525208, 0.0371929, -0.0897136, 0.0897136)
2: (-0.0562825, 0.1249244, -0.0562825, 0.1249244, -0.1812069, 0.1812069)
3: (-0.0198158, 0.0602108, -0.0198158, 0.0602108, -0.0800266, 0.0800266)
4: (-0.0541151, 0.0646047, -0.0541151, 0.0646047, -0.1187198, 0.1187198)
5: (-0.0397031, 0.0507637, -0.0397031, 0.0507637, -0.0904668, 0.0904668)
6: (-0.1079502, 0.0688725, -0.1079502, 0.0688725, -0.1768228, 0.1768228)
7: (0.8438617, 1.0119556, 0.8438617, 1.0119556, -0.1680939, 0.1680939)
8: (-0.0722072, 0.1026174, -0.0722072, 0.1026174, -0.1715834, 0.1715834)
9: (-0.0825652, 0.0632909, -0.0825652, 0.0632909, -0.1458561, 0.1458561)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.50 + 1.76 = 3.26 seconds
status: Status.VERIFIED
relational distance
Output dim: 7, lower bound: -0.1304819, upper bound: 0.1304819
