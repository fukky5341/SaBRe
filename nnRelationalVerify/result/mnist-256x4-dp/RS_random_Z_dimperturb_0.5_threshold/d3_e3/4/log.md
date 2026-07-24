## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0041507


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.9913323, 0.9982139, 0.9913323, 0.9982139, -0.0068681, 0.0068681)
1: (-0.0034237, -0.0017090, -0.0034237, -0.0017090, -0.0017113, 0.0017113)
2: (-0.0009972, 0.0080899, -0.0009972, 0.0080899, -0.0090692, 0.0090692)
3: (-0.0049553, -0.0008192, -0.0049553, -0.0008192, -0.0041279, 0.0041279)
4: (0.0003349, 0.0020937, 0.0003349, 0.0020937, -0.0017553, 0.0017553)
5: (-0.0022947, 0.0091345, -0.0022947, 0.0091345, -0.0114067, 0.0114067)
6: (-0.0007776, 0.0021233, -0.0007776, 0.0021233, -0.0028951, 0.0028951)
7: (-0.0051495, 0.0023559, -0.0051495, 0.0023559, -0.0074906, 0.0074906)
8: (-0.0022722, 0.0016748, -0.0022722, 0.0016748, -0.0039392, 0.0039392)
9: (-0.0038059, 0.0007709, -0.0038059, 0.0007709, -0.0045677, 0.0045677)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.54 + 2.00 = 3.54 seconds
status: Status.VERIFIED
relational distance
Output dim: 0, lower bound: -0.0038842, upper bound: 0.0038841
