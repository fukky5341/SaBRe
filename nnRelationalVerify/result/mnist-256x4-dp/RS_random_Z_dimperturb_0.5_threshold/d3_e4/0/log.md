## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.044602215


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0081818, 0.0009403, -0.0081818, 0.0009403, -0.0091221, 0.0091221)
1: (-0.0073852, 0.0021237, -0.0073852, 0.0021237, -0.0095088, 0.0095088)
2: (0.9504185, 0.9753396, 0.9504185, 0.9753396, -0.0249211, 0.0249211)
3: (0.0073361, 0.0495779, 0.0073361, 0.0495779, -0.0396362, 0.0396362)
4: (-0.0123261, 0.0201619, -0.0123261, 0.0201619, -0.0324879, 0.0324879)
5: (0.0088715, 0.0228163, 0.0088715, 0.0228163, -0.0139448, 0.0139448)
6: (-0.0088826, 0.0096368, -0.0088826, 0.0096368, -0.0185193, 0.0185193)
7: (-0.0291134, -0.0049753, -0.0291134, -0.0049753, -0.0241382, 0.0241382)
8: (-0.0056731, 0.0203303, -0.0056731, 0.0203303, -0.0260034, 0.0260034)
9: (-0.0123969, 0.0140017, -0.0123969, 0.0140017, -0.0263986, 0.0263986)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.68 + 1.81 = 3.49 seconds
status: Status.VERIFIED
relational distance
Output dim: 2, lower bound: -0.0197738, upper bound: 0.0197738
