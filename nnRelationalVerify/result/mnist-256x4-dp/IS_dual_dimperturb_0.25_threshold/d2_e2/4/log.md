## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0004263


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.9878166, 0.9891865, 0.9878166, 0.9891865, -0.0011994, 0.0011994)
1: (-0.0042997, -0.0039584, -0.0042997, -0.0039584, -0.0002989, 0.0002989)
2: (0.0109235, 0.0127323, 0.0109235, 0.0127323, -0.0015838, 0.0015838)
3: (-0.0070683, -0.0062450, -0.0070683, -0.0062450, -0.0007209, 0.0007209)
4: (0.0026421, 0.0029922, 0.0026421, 0.0029922, -0.0003065, 0.0003065)
5: (0.0126983, 0.0149734, 0.0126983, 0.0149734, -0.0019920, 0.0019920)
6: (-0.0022596, -0.0016821, -0.0022596, -0.0016821, -0.0005056, 0.0005056)
7: (-0.0089838, -0.0074898, -0.0089838, -0.0074898, -0.0013081, 0.0013081)
8: (-0.0042886, -0.0035030, -0.0042886, -0.0035030, -0.0006879, 0.0006879)
9: (0.0021980, 0.0031090, 0.0021980, 0.0031090, -0.0007977, 0.0007977)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.29 + 1.30 = 2.58 seconds
status: Status.ADV_EXAMPLE
