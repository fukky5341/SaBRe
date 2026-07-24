## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00010616


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0126843, -0.0118468, -0.0126843, -0.0118468, -0.0003709, 0.0003709)
1: (-0.0065148, -0.0062787, -0.0065148, -0.0062787, -0.0001046, 0.0001046)
2: (-0.0095082, -0.0077660, -0.0095082, -0.0077660, -0.0007716, 0.0007716)
3: (0.0003690, 0.0005996, 0.0003690, 0.0005996, -0.0001021, 0.0001021)
4: (0.0118957, 0.0131977, 0.0118957, 0.0131977, -0.0005767, 0.0005767)
5: (0.9988112, 0.9991730, 0.9988112, 0.9991730, -0.0001602, 0.0001602)
6: (0.0068046, 0.0071329, 0.0068046, 0.0071329, -0.0001454, 0.0001454)
7: (0.0020120, 0.0032373, 0.0020120, 0.0032373, -0.0005427, 0.0005427)
8: (-0.0117125, -0.0107588, -0.0117125, -0.0107588, -0.0004224, 0.0004224)
9: (-0.0030815, -0.0029992, -0.0030815, -0.0029992, -0.0000364, 0.0000364)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.75 + 1.23 = 2.97 seconds
status: Status.ADV_EXAMPLE
