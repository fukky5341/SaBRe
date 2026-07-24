## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0018212


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0002214, -0.0000739, -0.0002214, -0.0000739, -0.0001122, 0.0001122)
1: (0.0002836, 0.0009743, 0.0002836, 0.0009743, -0.0005251, 0.0005251)
2: (0.0148808, 0.0159152, 0.0148808, 0.0159152, -0.0007864, 0.0007864)
3: (0.0005628, 0.0013407, 0.0005628, 0.0013407, -0.0005913, 0.0005913)
4: (-0.0038605, -0.0031430, -0.0038605, -0.0031430, -0.0005455, 0.0005455)
5: (0.0085000, 0.0092764, 0.0085000, 0.0092764, -0.0005903, 0.0005903)
6: (0.0094323, 0.0097253, 0.0094323, 0.0097253, -0.0002227, 0.0002227)
7: (-0.0185375, -0.0168520, -0.0185375, -0.0168520, -0.0012814, 0.0012814)
8: (0.9706788, 0.9755080, 0.9706788, 0.9755080, -0.0036714, 0.0036714)
9: (0.0048514, 0.0062707, 0.0048514, 0.0062707, -0.0010790, 0.0010790)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.30 + 1.17 = 2.46 seconds
status: Status.ADV_EXAMPLE
