## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.002363328


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0003373, 0.0011271, 0.0003373, 0.0011271, -0.0007899, 0.0007899)
1: (0.9932824, 0.9953116, 0.9932824, 0.9953116, -0.0020292, 0.0020292)
2: (-0.0101109, -0.0028733, -0.0101109, -0.0028733, -0.0067813, 0.0067813)
3: (0.0031100, 0.0042491, 0.0031100, 0.0042491, -0.0011390, 0.0011390)
4: (0.0006879, 0.0064081, 0.0006879, 0.0064081, -0.0057202, 0.0057202)
5: (0.0041616, 0.0069752, 0.0041616, 0.0069752, -0.0028137, 0.0028137)
6: (-0.0026307, -0.0000211, -0.0026307, -0.0000211, -0.0026095, 0.0026095)
7: (-0.0087282, -0.0070747, -0.0087282, -0.0070747, -0.0016535, 0.0016535)
8: (0.0020268, 0.0115361, 0.0020268, 0.0115361, -0.0093864, 0.0093864)
9: (-0.0037097, -0.0016411, -0.0037097, -0.0016411, -0.0020686, 0.0020686)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.80 + 1.73 = 3.53 seconds
status: Status.VERIFIED
relational distance
Output dim: 1, lower bound: -0.0017565, upper bound: 0.0017564
