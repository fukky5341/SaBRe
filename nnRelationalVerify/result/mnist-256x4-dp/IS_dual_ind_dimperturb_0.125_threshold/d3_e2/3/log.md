## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00990792


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0111151, 0.0010882, -0.0111151, 0.0010882, -0.0097807, 0.0097807)
1: (-0.0118386, -0.0021185, -0.0118386, -0.0021185, -0.0094019, 0.0094019)
2: (0.0445417, 0.0496922, 0.0445417, 0.0496922, -0.0051505, 0.0051505)
3: (0.0093255, 0.0290059, 0.0093255, 0.0290059, -0.0154261, 0.0154261)
4: (-0.0036616, 0.0002438, -0.0036616, 0.0002438, -0.0039054, 0.0039054)
5: (0.0113448, 0.0134067, 0.0113448, 0.0134067, -0.0020619, 0.0020619)
6: (-0.0263778, -0.0144511, -0.0263778, -0.0144511, -0.0119267, 0.0119267)
7: (0.9203403, 0.9526963, 0.9203403, 0.9526963, -0.0323560, 0.0323560)
8: (0.0014786, 0.0164200, 0.0014786, 0.0164200, -0.0149414, 0.0149414)
9: (-0.0077917, -0.0031764, -0.0077917, -0.0031764, -0.0046153, 0.0046153)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.74 + 1.25 = 3.00 seconds
status: Status.ADV_EXAMPLE
