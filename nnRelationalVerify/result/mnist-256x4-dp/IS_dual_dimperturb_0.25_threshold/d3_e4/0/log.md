## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00357444


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0041046, -0.0036322, -0.0041046, -0.0036322, -0.0004724, 0.0004724)
1: (-0.0063724, -0.0042158, -0.0063724, -0.0042158, -0.0021566, 0.0021566)
2: (0.9688165, 0.9710761, 0.9688165, 0.9710761, -0.0022596, 0.0022596)
3: (0.0163006, 0.0329679, 0.0163006, 0.0329679, -0.0129226, 0.0129226)
4: (-0.0032004, -0.0019328, -0.0032004, -0.0019328, -0.0012676, 0.0012676)
5: (0.0136958, 0.0153169, 0.0136958, 0.0153169, -0.0016211, 0.0016211)
6: (0.0043297, 0.0050798, 0.0043297, 0.0050798, -0.0007501, 0.0007501)
7: (-0.0163222, -0.0120027, -0.0163222, -0.0120027, -0.0043195, 0.0043195)
8: (0.0037799, 0.0072068, 0.0037799, 0.0072068, -0.0034269, 0.0034269)
9: (0.0042234, 0.0106867, 0.0042234, 0.0106867, -0.0064633, 0.0064633)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.82 + 1.64 = 3.46 seconds
status: Status.VERIFIED
relational distance
Output dim: 2, lower bound: -0.0019044, upper bound: 0.0019044
