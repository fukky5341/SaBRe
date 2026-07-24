## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00010616


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0126834, -0.0117573, -0.0126834, -0.0117573, -0.0005514, 0.0005514)
1: (-0.0065146, -0.0062535, -0.0065146, -0.0062535, -0.0001555, 0.0001555)
2: (-0.0095061, -0.0075797, -0.0095061, -0.0075797, -0.0011470, 0.0011470)
3: (0.0003693, 0.0006242, 0.0003693, 0.0006242, -0.0001518, 0.0001518)
4: (0.0117564, 0.0131961, 0.0117564, 0.0131961, -0.0008572, 0.0008572)
5: (0.9987725, 0.9991725, 0.9987725, 0.9991725, -0.0002382, 0.0002382)
6: (0.0067695, 0.0071325, 0.0067695, 0.0071325, -0.0002162, 0.0002162)
7: (0.0018810, 0.0032359, 0.0018810, 0.0032359, -0.0008067, 0.0008067)
8: (-0.0117114, -0.0106568, -0.0117114, -0.0106568, -0.0006279, 0.0006279)
9: (-0.0030903, -0.0029993, -0.0030903, -0.0029993, -0.0000542, 0.0000542)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.78 + 1.21 = 3.00 seconds
status: Status.ADV_EXAMPLE
