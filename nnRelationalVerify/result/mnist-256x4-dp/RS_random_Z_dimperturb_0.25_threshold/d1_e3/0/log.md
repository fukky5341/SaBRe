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
Threshold: 0.000280174


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (1.0034041, 1.0042379, 1.0034041, 1.0042379, -0.0005276, 0.0005276)
1: (-0.0004157, -0.0002080, -0.0004157, -0.0002080, -0.0001315, 0.0001315)
2: (-0.0089518, -0.0078508, -0.0089518, -0.0078508, -0.0006967, 0.0006967)
3: (0.0023002, 0.0028013, 0.0023002, 0.0028013, -0.0003171, 0.0003171)
4: (-0.0012047, -0.0009916, -0.0012047, -0.0009916, -0.0001349, 0.0001349)
5: (-0.0122995, -0.0109148, -0.0122995, -0.0109148, -0.0008763, 0.0008763)
6: (0.0043111, 0.0046626, 0.0043111, 0.0046626, -0.0002224, 0.0002224)
7: (0.0080166, 0.0089259, 0.0080166, 0.0089259, -0.0005755, 0.0005755)
8: (0.0046517, 0.0051299, 0.0046517, 0.0051299, -0.0003026, 0.0003026)
9: (-0.0078122, -0.0072577, -0.0078122, -0.0072577, -0.0003509, 0.0003509)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.32 + 1.40 = 2.72 seconds
status: Status.VERIFIED
relational distance
Output dim: 0, lower bound: -0.0002565, upper bound: 0.0002566
