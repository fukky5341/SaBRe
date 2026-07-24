## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00117945


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0040978, -0.0038600, -0.0040978, -0.0038600, -0.0001577, 0.0001577)
1: (0.0011205, 0.0024373, 0.0011205, 0.0024373, -0.0008730, 0.0008730)
2: (0.0095209, 0.0124628, 0.0095209, 0.0124628, -0.0019503, 0.0019503)
3: (0.0020825, 0.0033222, 0.0020825, 0.0033222, -0.0008219, 0.0008219)
4: (1.0048295, 1.0096393, 1.0048295, 1.0096393, -0.0031885, 0.0031885)
5: (0.0031809, 0.0041166, 0.0031809, 0.0041166, -0.0006203, 0.0006203)
6: (-0.0111001, -0.0098825, -0.0111001, -0.0098825, -0.0008072, 0.0008072)
7: (-0.0102193, -0.0100640, -0.0102193, -0.0100640, -0.0001030, 0.0001030)
8: (-0.0040261, -0.0031848, -0.0040261, -0.0031848, -0.0005577, 0.0005577)
9: (-0.0022270, 0.0019847, -0.0022270, 0.0019847, -0.0027921, 0.0027921)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.63 + 1.27 = 2.90 seconds
status: Status.ADV_EXAMPLE
