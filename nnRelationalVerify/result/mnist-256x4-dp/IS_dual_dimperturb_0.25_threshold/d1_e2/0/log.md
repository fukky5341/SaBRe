## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.000280174


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0009393, 0.0011004, 0.0009393, 0.0011004, -0.0001578, 0.0001578)
1: (0.9936182, 0.9940932, 0.9936182, 0.9940932, -0.0004348, 0.0004348)
2: (-0.0065766, -0.0049929, -0.0065766, -0.0049929, -0.0014433, 0.0014433)
3: (0.0037937, 0.0040151, 0.0037937, 0.0040151, -0.0001942, 0.0001942)
4: (0.0023631, 0.0036148, 0.0023631, 0.0036148, -0.0012121, 0.0012121)
5: (0.0059930, 0.0065171, 0.0059930, 0.0065171, -0.0005241, 0.0005241)
6: (-0.0014039, -0.0008542, -0.0014039, -0.0008542, -0.0004966, 0.0004966)
7: (-0.0083274, -0.0079274, -0.0083274, -0.0079274, -0.0004000, 0.0004000)
8: (0.0048117, 0.0068925, 0.0048117, 0.0068925, -0.0016895, 0.0016895)
9: (-0.0036880, -0.0031875, -0.0036880, -0.0031875, -0.0005004, 0.0005004)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.16 + 1.29 = 2.45 seconds
status: Status.ADV_EXAMPLE
