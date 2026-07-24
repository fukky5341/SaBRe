## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00063364


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0043260, -0.0042120, -0.0043260, -0.0042120, -0.0000832, 0.0000832)
1: (0.0030699, 0.0037012, 0.0030699, 0.0037012, -0.0004609, 0.0004609)
2: (0.0066973, 0.0081076, 0.0066973, 0.0081076, -0.0010296, 0.0010296)
3: (0.0039178, 0.0045121, 0.0039178, 0.0045121, -0.0004339, 0.0004339)
4: (1.0119499, 1.0142555, 1.0119499, 1.0142555, -0.0016833, 0.0016833)
5: (0.0045661, 0.0050147, 0.0045661, 0.0050147, -0.0003275, 0.0003275)
6: (-0.0122688, -0.0116851, -0.0122688, -0.0116851, -0.0004262, 0.0004262)
7: (-0.0103684, -0.0102939, -0.0103684, -0.0102939, -0.0000544, 0.0000544)
8: (-0.0027806, -0.0023773, -0.0027806, -0.0023773, -0.0002944, 0.0002944)
9: (-0.0062693, -0.0042503, -0.0062693, -0.0042503, -0.0014740, 0.0014740)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.40 + 1.28 = 2.68 seconds
status: Status.ADV_EXAMPLE
