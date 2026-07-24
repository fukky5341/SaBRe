## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00013584


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0038690, -0.0025612, -0.0038690, -0.0025612, -0.0013078, 0.0013078)
1: (0.0055659, 0.0064770, 0.0055659, 0.0064770, -0.0009111, 0.0009111)
2: (0.0108435, 0.0133087, 0.0108435, 0.0133087, -0.0020655, 0.0020655)
3: (-0.0040256, -0.0029299, -0.0040256, -0.0029299, -0.0010927, 0.0010927)
4: (0.0047873, 0.0051533, 0.0047873, 0.0051533, -0.0002405, 0.0002405)
5: (-0.0018366, -0.0010160, -0.0018366, -0.0010160, -0.0008206, 0.0008206)
6: (-0.0057719, -0.0053757, -0.0057719, -0.0053757, -0.0003961, 0.0003961)
7: (-0.0031236, -0.0023802, -0.0031236, -0.0023802, -0.0007434, 0.0007434)
8: (-0.0032808, -0.0016483, -0.0032808, -0.0016483, -0.0016325, 0.0016325)
9: (1.0004487, 1.0006659, 1.0004487, 1.0006659, -0.0002172, 0.0002172)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.18 + 1.26 = 2.44 seconds
status: Status.ADV_EXAMPLE
