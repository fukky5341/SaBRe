## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 8.58e-05


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0041413, -0.0041340, -0.0041413, -0.0041340, -0.0000038, 0.0000038)
1: (-0.0077482, -0.0074751, -0.0077482, -0.0074751, -0.0001411, 0.0001411)
2: (0.9671653, 0.9674931, 0.9671653, 0.9674931, -0.0001694, 0.0001694)
3: (0.0041229, 0.0065397, 0.0041229, 0.0065397, -0.0012492, 0.0012492)
4: (-0.0011904, -0.0010066, -0.0011904, -0.0010066, -0.0000950, 0.0000950)
5: (0.0160672, 0.0162530, 0.0160672, 0.0162530, -0.0000960, 0.0000960)
6: (0.0040013, 0.0040917, 0.0040013, 0.0040917, -0.0000467, 0.0000467)
7: (-0.0094731, -0.0088467, -0.0094731, -0.0088467, -0.0003237, 0.0003237)
8: (0.0092136, 0.0097106, 0.0092136, 0.0097106, -0.0002568, 0.0002568)
9: (0.0142962, 0.0151900, 0.0142962, 0.0151900, -0.0004620, 0.0004620)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.55 + 1.21 = 2.76 seconds
status: Status.ADV_EXAMPLE
