## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.03662127


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.9667277, 0.9910905, 0.9667277, 0.9910905, -0.0243628, 0.0243628)
1: (-0.0047361, -0.0032608, -0.0047361, -0.0032608, -0.0014753, 0.0014753)
2: (0.0084091, 0.0150448, 0.0084091, 0.0150448, -0.0066358, 0.0066358)
3: (-0.0084414, -0.0051006, -0.0084414, -0.0051006, -0.0033409, 0.0033409)
4: (0.0021554, 0.0048262, 0.0021554, 0.0048262, -0.0026707, 0.0026707)
5: (0.0095359, 0.0322461, 0.0095359, 0.0322461, -0.0227102, 0.0227102)
6: (-0.0029978, -0.0005585, -0.0029978, -0.0005585, -0.0024393, 0.0024393)
7: (-0.0108938, -0.0054131, -0.0108938, -0.0054131, -0.0054807, 0.0054807)
8: (-0.0052931, 0.0020080, -0.0052931, 0.0020080, -0.0073011, 0.0073011)
9: (0.0009316, 0.0042737, 0.0009316, 0.0042737, -0.0033421, 0.0033421)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.84 + 1.87 = 3.72 seconds
status: Status.VERIFIED
relational distance
Output dim: 0, lower bound: -0.0210339, upper bound: 0.0210339
