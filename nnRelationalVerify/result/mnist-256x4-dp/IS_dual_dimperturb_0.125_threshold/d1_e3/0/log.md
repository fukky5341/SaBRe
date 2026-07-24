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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (1.0035753, 1.0040672, 1.0035753, 1.0040672, -0.0002806, 0.0002806)
1: (-0.0003731, -0.0002505, -0.0003731, -0.0002505, -0.0000699, 0.0000699)
2: (-0.0087264, -0.0080769, -0.0087264, -0.0080769, -0.0003705, 0.0003705)
3: (0.0024031, 0.0026988, 0.0024031, 0.0026988, -0.0001686, 0.0001686)
4: (-0.0011611, -0.0010354, -0.0011611, -0.0010354, -0.0000717, 0.0000717)
5: (-0.0120161, -0.0111991, -0.0120161, -0.0111991, -0.0004660, 0.0004660)
6: (0.0043833, 0.0045906, 0.0043833, 0.0045906, -0.0001183, 0.0001183)
7: (0.0082033, 0.0087398, 0.0082033, 0.0087398, -0.0003060, 0.0003060)
8: (0.0047499, 0.0050320, 0.0047499, 0.0050320, -0.0001609, 0.0001609)
9: (-0.0076987, -0.0073716, -0.0076987, -0.0073716, -0.0001866, 0.0001866)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.29 + 1.23 = 2.52 seconds
status: Status.VERIFIED
relational distance
Output dim: 0, lower bound: -0.0001316, upper bound: 0.0001318
