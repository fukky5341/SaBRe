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
0: (-0.0043318, -0.0042128, -0.0043318, -0.0042128, -0.0000821, 0.0000821)
1: (0.0030744, 0.0037329, 0.0030744, 0.0037329, -0.0004547, 0.0004547)
2: (0.0066264, 0.0080975, 0.0066264, 0.0080975, -0.0010158, 0.0010158)
3: (0.0039220, 0.0045420, 0.0039220, 0.0045420, -0.0004280, 0.0004280)
4: (1.0119663, 1.0143714, 1.0119663, 1.0143714, -0.0016607, 0.0016607)
5: (0.0045693, 0.0050372, 0.0045693, 0.0050372, -0.0003231, 0.0003231)
6: (-0.0122982, -0.0116893, -0.0122982, -0.0116893, -0.0004204, 0.0004204)
7: (-0.0103721, -0.0102944, -0.0103721, -0.0102944, -0.0000536, 0.0000536)
8: (-0.0027778, -0.0023571, -0.0027778, -0.0023571, -0.0002905, 0.0002905)
9: (-0.0063709, -0.0042648, -0.0063709, -0.0042648, -0.0014542, 0.0014542)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.27 + 1.23 = 2.50 seconds
status: Status.ADV_EXAMPLE
