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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0057970, 0.0099839, 0.0057970, 0.0099839, -0.0039138, 0.0039138)
1: (0.0016874, 0.0056760, 0.0016874, 0.0056760, -0.0038801, 0.0038801)
2: (-0.0209971, -0.0110150, -0.0209971, -0.0110150, -0.0068163, 0.0068163)
3: (-0.0048172, 0.0038450, -0.0048172, 0.0038450, -0.0075801, 0.0075801)
4: (0.0146447, 0.0160611, 0.0146447, 0.0160611, -0.0014164, 0.0014164)
5: (-0.0080330, 0.0041652, -0.0080330, 0.0041652, -0.0111145, 0.0111145)
6: (0.9920401, 1.0002590, 0.9920401, 1.0002590, -0.0071147, 0.0071147)
7: (0.0131266, 0.0172065, 0.0131266, 0.0172065, -0.0025008, 0.0025008)
8: (0.0035943, 0.0072922, 0.0035943, 0.0072922, -0.0036980, 0.0036980)
9: (-0.0235971, -0.0152113, -0.0235971, -0.0152113, -0.0064413, 0.0064414)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.66 + 1.59 = 3.25 seconds
status: Status.VERIFIED
relational distance
Output dim: 6, lower bound: -0.0036776, upper bound: 0.0036776
