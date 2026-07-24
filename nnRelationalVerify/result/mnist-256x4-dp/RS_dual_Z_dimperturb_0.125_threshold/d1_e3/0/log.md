## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.000149


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (1.0035495, 1.0040585, 1.0035495, 1.0040585, -0.0001378, 0.0001378)
1: (-0.0003795, -0.0002527, -0.0003795, -0.0002527, -0.0000343, 0.0000343)
2: (-0.0087149, -0.0080427, -0.0087149, -0.0080427, -0.0001819, 0.0001819)
3: (0.0023876, 0.0026935, 0.0023876, 0.0026935, -0.0000828, 0.0000828)
4: (-0.0011589, -0.0010288, -0.0011589, -0.0010288, -0.0000352, 0.0000352)
5: (-0.0120016, -0.0111561, -0.0120016, -0.0111561, -0.0002288, 0.0002288)
6: (0.0043724, 0.0045870, 0.0043724, 0.0045870, -0.0000581, 0.0000581)
7: (0.0081750, 0.0087303, 0.0081750, 0.0087303, -0.0001503, 0.0001503)
8: (0.0047350, 0.0050270, 0.0047350, 0.0050270, -0.0000790, 0.0000790)
9: (-0.0076929, -0.0073544, -0.0076929, -0.0073544, -0.0000916, 0.0000916)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.30 + 1.26 = 2.56 seconds
status: Status.VERIFIED
relational distance
Output dim: 0, lower bound: -0.0001378, upper bound: 0.0001378
