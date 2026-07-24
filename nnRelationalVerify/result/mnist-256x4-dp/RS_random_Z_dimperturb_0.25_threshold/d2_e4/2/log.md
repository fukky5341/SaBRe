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
0: (-0.0074534, 0.0023284, -0.0074534, 0.0023284, -0.0097818, 0.0097818)
1: (-0.0060617, -0.0005284, -0.0060617, -0.0005284, -0.0055334, 0.0055334)
2: (0.0284752, 0.0453710, 0.0284752, 0.0453710, -0.0168958, 0.0168958)
3: (-0.0036144, 0.0053698, -0.0036144, 0.0053698, -0.0089842, 0.0089842)
4: (-0.0086160, 0.0046403, -0.0086160, 0.0046403, -0.0132563, 0.0132563)
5: (0.0062060, 0.0178744, 0.0062060, 0.0178744, -0.0116684, 0.0116684)
6: (-0.0186338, 0.0054229, -0.0186338, 0.0054229, -0.0240568, 0.0240568)
7: (0.9627602, 0.9788862, 0.9627602, 0.9788862, -0.0161260, 0.0161260)
8: (-0.0231375, 0.0068974, -0.0231375, 0.0068974, -0.0300348, 0.0300348)
9: (-0.0079955, 0.0104706, -0.0079955, 0.0104706, -0.0184661, 0.0184661)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.59 + 1.54 = 3.12 seconds
status: Status.VERIFIED
relational distance
Output dim: 7, lower bound: -0.0118654, upper bound: 0.0118654
