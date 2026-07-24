## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00149824981


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0000148, 0.0011470, -0.0000148, 0.0011470, -0.0009517, 0.0009517)
1: (0.9934503, 0.9959105, 0.9934503, 0.9959105, -0.0020203, 0.0020203)
2: (-0.0080133, -0.0073141, -0.0080133, -0.0073141, -0.0005620, 0.0005620)
3: (0.0026695, 0.0041230, 0.0026695, 0.0041230, -0.0011942, 0.0011942)
4: (0.0024775, 0.0043720, 0.0024775, 0.0043720, -0.0016068, 0.0016068)
5: (0.0033272, 0.0060803, 0.0033272, 0.0060803, -0.0022409, 0.0022409)
6: (-0.0017954, 0.0007504, -0.0017954, 0.0007504, -0.0021070, 0.0021070)
7: (-0.0079029, -0.0067253, -0.0079029, -0.0067253, -0.0009527, 0.0009527)
8: (0.0078615, 0.0081926, 0.0078615, 0.0081926, -0.0002885, 0.0002885)
9: (-0.0037206, -0.0020390, -0.0037206, -0.0020390, -0.0013762, 0.0013762)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.34 + 1.50 = 2.84 seconds
status: Status.VERIFIED
relational distance
Output dim: 1, lower bound: -0.0013580, upper bound: 0.0013580
