## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0014131413


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0013196, 0.0006618, -0.0013196, 0.0006618, -0.0019814, 0.0019814)
1: (-0.0037888, -0.0026311, -0.0037888, -0.0026311, -0.0011577, 0.0011577)
2: (0.0320313, 0.0335031, 0.0320313, 0.0335031, -0.0014718, 0.0014718)
3: (-0.0027994, -0.0010804, -0.0027994, -0.0010804, -0.0017190, 0.0017190)
4: (-0.0027469, -0.0007763, -0.0027469, -0.0007763, -0.0019706, 0.0019706)
5: (0.0113057, 0.0131574, 0.0113057, 0.0131574, -0.0018517, 0.0018517)
6: (-0.0039586, -0.0014627, -0.0039586, -0.0014627, -0.0024960, 0.0024960)
7: (0.9756966, 0.9770358, 0.9756966, 0.9770358, -0.0013393, 0.0013393)
8: (-0.0126136, -0.0073283, -0.0126136, -0.0073283, -0.0052853, 0.0052853)
9: (0.0002390, 0.0033068, 0.0002390, 0.0033068, -0.0030679, 0.0030679)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.55 + 1.45 = 3.01 seconds
status: Status.VERIFIED
relational distance
Output dim: 7, lower bound: -0.0009634, upper bound: 0.0009634
