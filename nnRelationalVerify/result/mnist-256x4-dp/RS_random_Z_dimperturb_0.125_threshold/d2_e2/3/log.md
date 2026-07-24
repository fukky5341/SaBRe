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
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0002212, -0.0000725, -0.0002212, -0.0000725, -0.0001086, 0.0001086)
1: (0.0002848, 0.0009809, 0.0002848, 0.0009809, -0.0005085, 0.0005085)
2: (0.0148710, 0.0159134, 0.0148710, 0.0159134, -0.0007616, 0.0007616)
3: (0.0005554, 0.0013393, 0.0005554, 0.0013393, -0.0005727, 0.0005727)
4: (-0.0038673, -0.0031442, -0.0038673, -0.0031442, -0.0005282, 0.0005282)
5: (0.0084926, 0.0092751, 0.0084926, 0.0092751, -0.0005716, 0.0005716)
6: (0.0094328, 0.0097280, 0.0094328, 0.0097280, -0.0002157, 0.0002157)
7: (-0.0185346, -0.0168360, -0.0185346, -0.0168360, -0.0012409, 0.0012409)
8: (0.9706872, 0.9755539, 0.9706872, 0.9755539, -0.0035554, 0.0035554)
9: (0.0048379, 0.0062682, 0.0048379, 0.0062682, -0.0010449, 0.0010449)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.29 + 1.19 = 2.48 seconds
status: Status.ADV_EXAMPLE
