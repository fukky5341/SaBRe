## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00146637


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0040753, -0.0037757, -0.0040753, -0.0037757, -0.0002996, 0.0002996)
1: (-0.0052770, -0.0043623, -0.0052770, -0.0043623, -0.0009147, 0.0009147)
2: (0.9697240, 0.9710107, 0.9697240, 0.9710107, -0.0012867, 0.0012867)
3: (0.0259964, 0.0324861, 0.0259964, 0.0324861, -0.0047542, 0.0047542)
4: (-0.0031638, -0.0025867, -0.0031638, -0.0025867, -0.0005771, 0.0005771)
5: (0.0138473, 0.0145716, 0.0138473, 0.0145716, -0.0007243, 0.0007243)
6: (0.0045642, 0.0050618, 0.0045642, 0.0050618, -0.0004976, 0.0004976)
7: (-0.0161973, -0.0145154, -0.0161973, -0.0145154, -0.0016819, 0.0016819)
8: (0.0038790, 0.0052133, 0.0038790, 0.0052133, -0.0013343, 0.0013343)
9: (0.0045025, 0.0071012, 0.0045025, 0.0071012, -0.0025987, 0.0025987)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.34 + 1.46 = 2.80 seconds
status: Status.VERIFIED
relational distance
Output dim: 2, lower bound: -0.0011579, upper bound: 0.0011580
