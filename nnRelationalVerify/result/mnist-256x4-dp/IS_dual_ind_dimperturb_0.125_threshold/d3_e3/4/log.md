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
Threshold: 0.00027306


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.9945356, 0.9960137, 0.9945356, 0.9960137, -0.0009872, 0.0009872)
1: (-0.0026255, -0.0022572, -0.0026255, -0.0022572, -0.0002460, 0.0002460)
2: (0.0019081, 0.0038599, 0.0019081, 0.0038599, -0.0013036, 0.0013036)
3: (-0.0030300, -0.0021416, -0.0030300, -0.0021416, -0.0005934, 0.0005934)
4: (0.0008972, 0.0012750, 0.0008972, 0.0012750, -0.0002523, 0.0002523)
5: (0.0013593, 0.0038142, 0.0013593, 0.0038142, -0.0016396, 0.0016396)
6: (0.0005727, 0.0011958, 0.0005727, 0.0011958, -0.0004162, 0.0004162)
7: (-0.0016558, -0.0000437, -0.0016558, -0.0000437, -0.0010767, 0.0010767)
8: (-0.0004349, 0.0004129, -0.0004349, 0.0004129, -0.0005662, 0.0005662)
9: (-0.0023426, -0.0013596, -0.0023426, -0.0013596, -0.0006566, 0.0006566)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.73 + 1.26 = 2.99 seconds
status: Status.ADV_EXAMPLE
