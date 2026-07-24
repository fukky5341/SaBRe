## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00473928


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0095453, -0.0030190, -0.0095453, -0.0030190, -0.0039925, 0.0039925)
1: (-0.0050512, -0.0043051, -0.0050512, -0.0043051, -0.0007461, 0.0007461)
2: (0.0325273, 0.0494672, 0.0325273, 0.0494672, -0.0112865, 0.0112865)
3: (0.0005608, 0.0111271, 0.0005608, 0.0111271, -0.0050098, 0.0050098)
4: (-0.0041482, -0.0021966, -0.0041482, -0.0021966, -0.0019517, 0.0019517)
5: (0.0100430, 0.0118735, 0.0100430, 0.0118735, -0.0018305, 0.0018305)
6: (-0.0171635, -0.0019270, -0.0171635, -0.0019270, -0.0075751, 0.0075751)
7: (0.9543850, 0.9752446, 0.9543850, 0.9752446, -0.0208597, 0.0208597)
8: (-0.0056339, 0.0007332, -0.0056339, 0.0007332, -0.0063671, 0.0063671)
9: (-0.0041716, -0.0009381, -0.0041716, -0.0009381, -0.0032335, 0.0032335)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.43 + 1.27 = 2.70 seconds
status: Status.ADV_EXAMPLE
