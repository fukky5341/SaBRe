## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00226296


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0021964, -0.0006448, -0.0021964, -0.0006448, -0.0015516, 0.0015516)
1: (0.9980873, 1.0013763, 0.9980873, 1.0013763, -0.0024590, 0.0024590)
2: (-0.0015545, 0.0011366, -0.0015545, 0.0011366, -0.0026391, 0.0026391)
3: (0.0010995, 0.0021370, 0.0010995, 0.0021370, -0.0007574, 0.0007574)
4: (-0.0010367, 0.0008792, -0.0010367, 0.0008792, -0.0017128, 0.0017128)
5: (-0.0002206, 0.0021110, -0.0002206, 0.0021110, -0.0023315, 0.0023315)
6: (0.0000328, 0.0012397, 0.0000328, 0.0012397, -0.0012069, 0.0012069)
7: (-0.0051321, -0.0031603, -0.0051321, -0.0031603, -0.0016048, 0.0016048)
8: (-0.0089721, -0.0037822, -0.0089721, -0.0037822, -0.0044455, 0.0044455)
9: (0.0018819, 0.0049265, 0.0018819, 0.0049265, -0.0030446, 0.0030446)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.65 + 1.58 = 3.22 seconds
status: Status.VERIFIED
relational distance
Output dim: 1, lower bound: -0.0019420, upper bound: 0.0019419
