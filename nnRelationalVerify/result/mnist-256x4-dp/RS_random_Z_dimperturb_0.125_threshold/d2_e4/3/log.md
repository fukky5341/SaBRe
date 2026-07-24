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
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0023390, -0.0006421, -0.0023390, -0.0006421, -0.0016969, 0.0016969)
1: (0.9980806, 1.0016596, 0.9980806, 1.0016596, -0.0026960, 0.0026960)
2: (-0.0015609, 0.0012844, -0.0015609, 0.0012844, -0.0028453, 0.0028453)
3: (0.0010596, 0.0021398, 0.0010596, 0.0021398, -0.0007644, 0.0007644)
4: (-0.0012245, 0.0008823, -0.0012245, 0.0008823, -0.0019481, 0.0019481)
5: (-0.0002856, 0.0021175, -0.0002856, 0.0021175, -0.0024031, 0.0024031)
6: (-0.0002573, 0.0012627, -0.0002573, 0.0012627, -0.0015199, 0.0015199)
7: (-0.0051368, -0.0030568, -0.0051368, -0.0030568, -0.0017221, 0.0017221)
8: (-0.0091168, -0.0037676, -0.0091168, -0.0037676, -0.0045862, 0.0045862)
9: (0.0018733, 0.0050114, 0.0018733, 0.0050114, -0.0031381, 0.0031381)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.69 + 1.57 = 3.25 seconds
status: Status.VERIFIED
relational distance
Output dim: 1, lower bound: -0.0022191, upper bound: 0.0022191
