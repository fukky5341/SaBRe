## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00061831


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0039672, -0.0038292, -0.0039672, -0.0038292, -0.0000508, 0.0000508)
1: (0.0009504, 0.0017142, 0.0009504, 0.0017142, -0.0002814, 0.0002814)
2: (0.0111365, 0.0128429, 0.0111365, 0.0128429, -0.0006287, 0.0006287)
3: (0.0019223, 0.0026414, 0.0019223, 0.0026414, -0.0002650, 0.0002650)
4: (1.0042082, 1.0069978, 1.0042082, 1.0069978, -0.0010279, 0.0010279)
5: (0.0030601, 0.0036028, 0.0030601, 0.0036028, -0.0002000, 0.0002000)
6: (-0.0104314, -0.0097252, -0.0104314, -0.0097252, -0.0002602, 0.0002602)
7: (-0.0101340, -0.0100439, -0.0101340, -0.0100439, -0.0000332, 0.0000332)
8: (-0.0041348, -0.0036468, -0.0041348, -0.0036468, -0.0001798, 0.0001798)
9: (0.0000860, 0.0025288, 0.0000860, 0.0025288, -0.0009001, 0.0009001)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.96 + 1.27 = 3.23 seconds
status: Status.ADV_EXAMPLE
