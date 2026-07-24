## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00104286


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0166680, 0.0176580, 0.0166680, 0.0176580, -0.0006240, 0.0006240)
1: (-0.0007747, -0.0000694, -0.0007747, -0.0000694, -0.0004576, 0.0004576)
2: (0.0037617, 0.0040823, 0.0037617, 0.0040823, -0.0001997, 0.0001997)
3: (0.0016375, 0.0022452, 0.0016375, 0.0022452, -0.0003265, 0.0003265)
4: (-0.0042123, -0.0034007, -0.0042123, -0.0034007, -0.0004089, 0.0004089)
5: (-0.0001086, 0.0003193, -0.0001086, 0.0003193, -0.0002796, 0.0002796)
6: (-0.0041509, -0.0025847, -0.0041509, -0.0025847, -0.0007502, 0.0007502)
7: (-0.0204153, -0.0157812, -0.0204153, -0.0157812, -0.0023573, 0.0023573)
8: (0.9766860, 0.9807488, 0.9766860, 0.9807488, -0.0021727, 0.0021727)
9: (0.0026771, 0.0056941, 0.0026771, 0.0056941, -0.0015499, 0.0015499)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.27 + 1.23 = 2.49 seconds
status: Status.ADV_EXAMPLE
