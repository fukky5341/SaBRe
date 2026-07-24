## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00017725


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0041455, -0.0041303, -0.0041455, -0.0041303, -0.0000121, 0.0000121)
1: (-0.0079041, -0.0073362, -0.0079041, -0.0073362, -0.0004535, 0.0004535)
2: (0.9669783, 0.9676597, 0.9669783, 0.9676597, -0.0005442, 0.0005442)
3: (0.0027427, 0.0077692, 0.0027427, 0.0077692, -0.0040140, 0.0040140)
4: (-0.0012839, -0.0009016, -0.0012839, -0.0009016, -0.0003053, 0.0003053)
5: (0.0159727, 0.0163591, 0.0159727, 0.0163591, -0.0003085, 0.0003085)
6: (0.0039497, 0.0041377, 0.0039497, 0.0041377, -0.0001501, 0.0001501)
7: (-0.0097917, -0.0084891, -0.0097917, -0.0084891, -0.0010403, 0.0010403)
8: (0.0089609, 0.0099943, 0.0089609, 0.0099943, -0.0008253, 0.0008253)
9: (0.0138416, 0.0157004, 0.0138416, 0.0157004, -0.0014844, 0.0014844)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.48 + 1.28 = 2.76 seconds
status: Status.ADV_EXAMPLE
