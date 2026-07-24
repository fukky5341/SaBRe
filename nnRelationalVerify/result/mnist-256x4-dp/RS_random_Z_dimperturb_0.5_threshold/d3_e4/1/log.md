## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.1285293


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0319368, 0.0315115, -0.0319368, 0.0315115, -0.0634483, 0.0634483)
1: (-0.0311957, 0.0260643, -0.0311957, 0.0260643, -0.0572600, 0.0572600)
2: (-0.0262381, 0.1109071, -0.0262381, 0.1109071, -0.1371453, 0.1371453)
3: (-0.0151548, 0.0415155, -0.0151548, 0.0415155, -0.0566703, 0.0566703)
4: (-0.0418257, 0.0363284, -0.0418257, 0.0363284, -0.0781541, 0.0781541)
5: (-0.0175112, 0.0468639, -0.0175112, 0.0468639, -0.0643751, 0.0643751)
6: (-0.0795149, 0.0506542, -0.0795149, 0.0506542, -0.1301691, 0.1301691)
7: (0.8745943, 0.9975381, 0.8745943, 0.9975381, -0.1229438, 0.1229438)
8: (-0.0688164, 0.0728520, -0.0688164, 0.0728520, -0.1416684, 0.1416684)
9: (-0.0604489, 0.0595953, -0.0604489, 0.0595953, -0.1200442, 0.1200442)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.80 + 2.02 = 3.83 seconds
status: Status.VERIFIED
relational distance
Output dim: 7, lower bound: -0.1015411, upper bound: 0.1015411
