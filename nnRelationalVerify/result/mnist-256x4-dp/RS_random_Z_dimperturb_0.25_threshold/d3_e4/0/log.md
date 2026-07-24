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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0041044, -0.0033117, -0.0041044, -0.0033117, -0.0007927, 0.0007927)
1: (-0.0063642, -0.0038887, -0.0063642, -0.0038887, -0.0024755, 0.0024755)
2: (0.9678245, 0.9712219, 0.9678245, 0.9712219, -0.0033975, 0.0033975)
3: (0.0163732, 0.0340435, 0.0163732, 0.0340435, -0.0139820, 0.0139820)
4: (-0.0032822, -0.0017585, -0.0032822, -0.0017585, -0.0015237, 0.0015237)
5: (0.0133577, 0.0153113, 0.0133577, 0.0153113, -0.0019536, 0.0019536)
6: (0.0038062, 0.0051200, 0.0038062, 0.0051200, -0.0013138, 0.0013138)
7: (-0.0166009, -0.0120215, -0.0166009, -0.0120215, -0.0045795, 0.0045795)
8: (0.0035588, 0.0071918, 0.0035588, 0.0071918, -0.0036331, 0.0036331)
9: (0.0036004, 0.0106598, 0.0036004, 0.0106598, -0.0070595, 0.0070595)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.85 + 1.68 = 3.53 seconds
status: Status.VERIFIED
relational distance
Output dim: 2, lower bound: -0.0030482, upper bound: 0.0030482
