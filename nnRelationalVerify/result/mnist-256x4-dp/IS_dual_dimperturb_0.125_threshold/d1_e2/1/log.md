## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 5.355e-05


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0037285, -0.0029844, -0.0037285, -0.0029844, -0.0007441, 0.0007441)
1: (0.0058654, 0.0063713, 0.0058654, 0.0063713, -0.0005059, 0.0005059)
2: (0.0111083, 0.0125115, 0.0111083, 0.0125115, -0.0011519, 0.0011519)
3: (-0.0036684, -0.0030523, -0.0036684, -0.0030523, -0.0006124, 0.0006124)
4: (0.0049063, 0.0051131, 0.0049063, 0.0051131, -0.0001155, 0.0001155)
5: (-0.0015710, -0.0011037, -0.0015710, -0.0011037, -0.0004673, 0.0004673)
6: (-0.0056436, -0.0054186, -0.0056436, -0.0054186, -0.0002249, 0.0002249)
7: (-0.0030195, -0.0026343, -0.0030195, -0.0026343, -0.0003852, 0.0003852)
8: (-0.0027530, -0.0018240, -0.0027530, -0.0018240, -0.0009290, 0.0009290)
9: (1.0004641, 1.0005312, 1.0004641, 1.0005312, -0.0000671, 0.0000671)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.36 + 1.24 = 2.60 seconds
status: Status.VERIFIED
relational distance
Output dim: 9, lower bound: -0.0000454, upper bound: 0.0000454
