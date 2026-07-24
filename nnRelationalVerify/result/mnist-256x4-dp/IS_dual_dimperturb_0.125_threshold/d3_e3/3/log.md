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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0126576, -0.0119656, -0.0126576, -0.0119656, -0.0003379, 0.0003379)
1: (-0.0065073, -0.0063122, -0.0065073, -0.0063122, -0.0000953, 0.0000953)
2: (-0.0094524, -0.0080129, -0.0094524, -0.0080129, -0.0007029, 0.0007029)
3: (0.0003764, 0.0005669, 0.0003764, 0.0005669, -0.0000930, 0.0000930)
4: (0.0120802, 0.0131560, 0.0120802, 0.0131560, -0.0005253, 0.0005253)
5: (0.9988625, 0.9991614, 0.9988625, 0.9991614, -0.0001459, 0.0001459)
6: (0.0068511, 0.0071224, 0.0068511, 0.0071224, -0.0001325, 0.0001325)
7: (0.0021857, 0.0031982, 0.0021857, 0.0031982, -0.0004944, 0.0004944)
8: (-0.0116820, -0.0108940, -0.0116820, -0.0108940, -0.0003848, 0.0003848)
9: (-0.0030699, -0.0030019, -0.0030699, -0.0030019, -0.0000332, 0.0000332)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.92 + 1.33 = 3.25 seconds
status: Status.VERIFIED
relational distance
Output dim: 5, lower bound: -0.0001059, upper bound: 0.0001059
