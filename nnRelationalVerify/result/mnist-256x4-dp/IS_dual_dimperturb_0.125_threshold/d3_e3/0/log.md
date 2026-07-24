## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00046665


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0026587, -0.0021587, -0.0026587, -0.0021587, -0.0002526, 0.0002526)
1: (-0.0110575, -0.0097886, -0.0110575, -0.0097886, -0.0006410, 0.0006410)
2: (0.0281699, 0.0289571, 0.0281699, 0.0289571, -0.0003977, 0.0003977)
3: (0.0054922, 0.0069622, 0.0054922, 0.0069622, -0.0007426, 0.0007426)
4: (-0.0101404, -0.0088497, -0.0101404, -0.0088497, -0.0006520, 0.0006520)
5: (0.0098973, 0.0103862, 0.0098973, 0.0103862, -0.0002470, 0.0002470)
6: (0.0073584, 0.0092240, 0.0073584, 0.0092240, -0.0009424, 0.0009424)
7: (0.9832083, 0.9845138, 0.9832083, 0.9845138, -0.0006595, 0.0006595)
8: (-0.0045675, -0.0031679, -0.0045675, -0.0031679, -0.0007070, 0.0007070)
9: (-0.0029070, -0.0019825, -0.0029070, -0.0019825, -0.0004670, 0.0004670)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.78 + 1.35 = 3.12 seconds
status: Status.VERIFIED
relational distance
Output dim: 7, lower bound: -0.0004449, upper bound: 0.0004449
