## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.01412451


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0084710, 0.0026976, -0.0084710, 0.0026976, -0.0111686, 0.0111686)
1: (-0.0064648, -0.0000861, -0.0064648, -0.0000861, -0.0063787, 0.0063787)
2: (0.0278853, 0.0477508, 0.0278853, 0.0477508, -0.0198655, 0.0198655)
3: (-0.0037496, 0.0064733, -0.0037496, 0.0064733, -0.0102229, 0.0102229)
4: (-0.0097597, 0.0056981, -0.0097597, 0.0056981, -0.0154578, 0.0154578)
5: (0.0053600, 0.0189082, 0.0053600, 0.0189082, -0.0135482, 0.0135482)
6: (-0.0218245, 0.0068163, -0.0218245, 0.0068163, -0.0286407, 0.0286407)
7: (0.9602942, 0.9792315, 0.9602942, 0.9792315, -0.0189372, 0.0189372)
8: (-0.0250291, 0.0092573, -0.0250291, 0.0092573, -0.0342864, 0.0342864)
9: (-0.0093615, 0.0118549, -0.0093615, 0.0118549, -0.0212165, 0.0212165)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.54 + 1.47 = 3.01 seconds
status: Status.VERIFIED
relational distance
Output dim: 7, lower bound: -0.0138457, upper bound: 0.0138457
