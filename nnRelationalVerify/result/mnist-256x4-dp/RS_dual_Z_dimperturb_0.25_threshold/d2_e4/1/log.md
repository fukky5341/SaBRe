## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00372996


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0056557, 0.0098620, 0.0056557, 0.0098620, -0.0037595, 0.0037595)
1: (0.0015529, 0.0055599, 0.0015529, 0.0055599, -0.0037261, 0.0037261)
2: (-0.0213339, -0.0113054, -0.0213339, -0.0113054, -0.0063931, 0.0063931)
3: (-0.0045652, 0.0041372, -0.0045652, 0.0041372, -0.0072734, 0.0072734)
4: (0.0147207, 0.0160397, 0.0147207, 0.0160397, -0.0013190, 0.0013190)
5: (-0.0076780, 0.0045767, -0.0076780, 0.0045767, -0.0106753, 0.0106753)
6: (0.9922793, 1.0005363, 0.9922793, 1.0005363, -0.0068242, 0.0068242)
7: (0.0132641, 0.0173282, 0.0132641, 0.0173282, -0.0022520, 0.0022520)
8: (0.0034695, 0.0071846, 0.0034695, 0.0071846, -0.0037152, 0.0037152)
9: (-0.0238799, -0.0154554, -0.0238799, -0.0154554, -0.0061293, 0.0061293)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.60 + 1.60 = 3.19 seconds
status: Status.VERIFIED
relational distance
Output dim: 6, lower bound: -0.0035600, upper bound: 0.0035601
