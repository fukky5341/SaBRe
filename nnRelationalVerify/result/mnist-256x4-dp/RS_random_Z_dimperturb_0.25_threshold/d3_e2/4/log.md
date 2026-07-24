## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00117945


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0040980, -0.0038871, -0.0040980, -0.0038871, -0.0001525, 0.0001525)
1: (0.0012709, 0.0024386, 0.0012709, 0.0024386, -0.0008445, 0.0008445)
2: (0.0095180, 0.0121267, 0.0095180, 0.0121267, -0.0018866, 0.0018866)
3: (0.0022241, 0.0033235, 0.0022241, 0.0033235, -0.0007950, 0.0007950)
4: (1.0053790, 1.0096440, 1.0053790, 1.0096440, -0.0030844, 0.0030844)
5: (0.0032878, 0.0041176, 0.0032878, 0.0041176, -0.0006000, 0.0006000)
6: (-0.0111013, -0.0100216, -0.0111013, -0.0100216, -0.0007809, 0.0007809)
7: (-0.0102194, -0.0100817, -0.0102194, -0.0100817, -0.0000996, 0.0000996)
8: (-0.0039300, -0.0031840, -0.0039300, -0.0031840, -0.0005395, 0.0005395)
9: (-0.0022312, 0.0015036, -0.0022312, 0.0015036, -0.0027009, 0.0027009)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.57 + 1.29 = 2.86 seconds
status: Status.ADV_EXAMPLE
