## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 5.708e-05


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0041694, -0.0041626, -0.0041694, -0.0041626, -0.0000032, 0.0000032)
1: (-0.0088005, -0.0085468, -0.0088005, -0.0085468, -0.0001192, 0.0001192)
2: (0.9659024, 0.9662070, 0.9659024, 0.9662070, -0.0001430, 0.0001430)
3: (-0.0051919, -0.0029456, -0.0051919, -0.0029456, -0.0010551, 0.0010551)
4: (-0.0004690, -0.0002982, -0.0004690, -0.0002982, -0.0000802, 0.0000802)
5: (0.0167963, 0.0169690, 0.0167963, 0.0169690, -0.0000811, 0.0000811)
6: (0.0036531, 0.0037371, 0.0036531, 0.0037371, -0.0000394, 0.0000394)
7: (-0.0070149, -0.0064327, -0.0070149, -0.0064327, -0.0002734, 0.0002734)
8: (0.0111639, 0.0116257, 0.0111639, 0.0116257, -0.0002169, 0.0002169)
9: (0.0178039, 0.0186346, 0.0178039, 0.0186346, -0.0003902, 0.0003902)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.28 + 1.16 = 2.44 seconds
status: Status.ADV_EXAMPLE
