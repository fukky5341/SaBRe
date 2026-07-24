## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00147475


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0003672, 0.0026107, 0.0003672, 0.0026107, -0.0013500, 0.0013500)
1: (0.0013753, 0.0016995, 0.0013753, 0.0016995, -0.0001950, 0.0001950)
2: (0.0129165, 0.0141569, 0.0129165, 0.0141569, -0.0007464, 0.0007464)
3: (-0.0013216, -0.0000388, -0.0013216, -0.0000388, -0.0007719, 0.0007719)
4: (-0.0039950, -0.0026062, -0.0039950, -0.0026062, -0.0008357, 0.0008357)
5: (0.0065797, 0.0078940, 0.0065797, 0.0078940, -0.0007908, 0.0007908)
6: (0.0038060, 0.0090206, 0.0038060, 0.0090206, -0.0031377, 0.0031377)
7: (-0.0148420, -0.0077401, -0.0148420, -0.0077401, -0.0042733, 0.0042733)
8: (0.9787588, 0.9837615, 0.9787588, 0.9837615, -0.0030102, 0.0030102)
9: (-0.0011471, 0.0033940, -0.0011471, 0.0033940, -0.0027325, 0.0027325)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.56 + 1.28 = 2.84 seconds
status: Status.ADV_EXAMPLE
