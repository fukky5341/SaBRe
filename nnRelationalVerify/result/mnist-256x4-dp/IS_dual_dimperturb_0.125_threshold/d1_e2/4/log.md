## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 8.752e-05


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0041960, -0.0041914, -0.0041960, -0.0041914, -0.0000024, 0.0000024)
1: (-0.0097952, -0.0096243, -0.0097952, -0.0096243, -0.0000885, 0.0000885)
2: (0.9647088, 0.9649138, 0.9647088, 0.9649138, -0.0001062, 0.0001062)
3: (-0.0139962, -0.0124836, -0.0139962, -0.0124836, -0.0007833, 0.0007833)
4: (0.0002564, 0.0003715, 0.0002564, 0.0003715, -0.0000596, 0.0000596)
5: (0.0175295, 0.0176458, 0.0175295, 0.0176458, -0.0000602, 0.0000602)
6: (0.0033239, 0.0033804, 0.0033239, 0.0033804, -0.0000293, 0.0000293)
7: (-0.0045430, -0.0041510, -0.0045430, -0.0041510, -0.0002030, 0.0002030)
8: (0.0131249, 0.0134359, 0.0131249, 0.0134359, -0.0001610, 0.0001610)
9: (0.0213311, 0.0218904, 0.0213311, 0.0218904, -0.0002897, 0.0002897)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.27 + 1.18 = 2.45 seconds
status: Status.ADV_EXAMPLE
