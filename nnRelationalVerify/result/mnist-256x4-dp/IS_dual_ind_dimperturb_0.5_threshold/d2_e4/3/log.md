## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.40684923


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.1619782, 0.1989716, -0.1619782, 0.1989716, -0.3609498, 0.3609498)
1: (0.6709704, 1.0646353, 0.6709704, 1.0646353, -0.3936648, 0.3936648)
2: (-0.1290442, 0.1797831, -0.1290442, 0.1797831, -0.3088273, 0.3088273)
3: (-0.0841855, 0.1283819, -0.0841855, 0.1283819, -0.2125674, 0.2125674)
4: (-0.1362261, 0.1252102, -0.1362261, 0.1252102, -0.2614363, 0.2614363)
5: (-0.1316006, 0.1433123, -0.1316006, 0.1433123, -0.2749129, 0.2749129)
6: (-0.1781695, 0.1543939, -0.1781695, 0.1543939, -0.3325635, 0.3325635)
7: (-0.1307067, 0.1705325, -0.1307067, 0.1705325, -0.3012392, 0.3012392)
8: (-0.0863418, 0.2339270, -0.0863418, 0.2339270, -0.3202688, 0.3202688)
9: (-0.1622172, 0.1670155, -0.1622172, 0.1670155, -0.3292328, 0.3292328)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.52 + 2.12 = 3.64 seconds
status: Status.VERIFIED
relational distance
Output dim: 1, lower bound: -0.3449586, upper bound: 0.3449586
