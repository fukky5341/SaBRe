## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00990792


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0124354, 0.0034241, -0.0124354, 0.0034241, -0.0130179, 0.0130179)
1: (-0.0135172, -0.0008723, -0.0135172, -0.0008723, -0.0122750, 0.0122750)
2: (0.0438609, 0.0510941, 0.0438609, 0.0510941, -0.0072332, 0.0072332)
3: (0.0069643, 0.0325174, 0.0069643, 0.0325174, -0.0205334, 0.0205334)
4: (-0.0044614, 0.0007587, -0.0044614, 0.0007587, -0.0052201, 0.0052201)
5: (0.0107069, 0.0137167, 0.0107069, 0.0137167, -0.0030098, 0.0030098)
6: (-0.0290935, -0.0128635, -0.0290935, -0.0128635, -0.0162300, 0.0162300)
7: (0.9103597, 0.9566027, 0.9103597, 0.9566027, -0.0462430, 0.0462430)
8: (-0.0006788, 0.0201192, -0.0006788, 0.0201192, -0.0207980, 0.0207980)
9: (-0.0096388, -0.0016077, -0.0096388, -0.0016077, -0.0080311, 0.0080311)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.61 + 1.26 = 2.87 seconds
status: Status.ADV_EXAMPLE
