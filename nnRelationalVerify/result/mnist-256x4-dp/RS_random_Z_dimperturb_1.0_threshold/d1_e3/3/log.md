## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.09862356593824724


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0166849, 0.0250697, -0.0166849, 0.0250697, -0.0417546, 0.0417546)
1: (-0.0139894, 0.0534196, -0.0139894, 0.0534196, -0.0674090, 0.0674090)
2: (0.0022807, 0.0438471, 0.0022807, 0.0438471, -0.0415664, 0.0415664)
3: (-0.0161741, 0.0294320, -0.0161741, 0.0294320, -0.0456061, 0.0456061)
4: (-0.0388691, 0.0247383, -0.0388691, 0.0247383, -0.0636074, 0.0636074)
5: (-0.0199401, 0.0418346, -0.0199401, 0.0418346, -0.0617746, 0.0617746)
6: (-0.0128523, 0.0287888, -0.0128523, 0.0287888, -0.0416411, 0.0416411)
7: (-0.0376090, 0.0265432, -0.0376090, 0.0265432, -0.0641521, 0.0641521)
8: (-0.0134290, 0.0488630, -0.0134290, 0.0488630, -0.0622921, 0.0622921)
9: (0.8556051, 1.0088843, 0.8556051, 1.0088843, -0.1532792, 0.1532792)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.18 + 3.14 = 4.32 seconds
status: Status.VERIFIED
relational distance
Output dim: 9, lower bound: -0.0882511, upper bound: 0.0882511
