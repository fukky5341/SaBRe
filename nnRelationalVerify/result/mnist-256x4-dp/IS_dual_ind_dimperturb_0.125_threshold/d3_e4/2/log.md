## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00061831


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0039632, -0.0038002, -0.0039632, -0.0038002, -0.0000687, 0.0000687)
1: (0.0007896, 0.0016920, 0.0007896, 0.0016920, -0.0003804, 0.0003804)
2: (0.0111860, 0.0132022, 0.0111860, 0.0132022, -0.0008498, 0.0008498)
3: (0.0017709, 0.0026205, 0.0017709, 0.0026205, -0.0003581, 0.0003581)
4: (1.0036207, 1.0069170, 1.0036207, 1.0069170, -0.0013894, 0.0013894)
5: (0.0029458, 0.0035870, 0.0029458, 0.0035870, -0.0002703, 0.0002703)
6: (-0.0104109, -0.0095765, -0.0104109, -0.0095765, -0.0003517, 0.0003517)
7: (-0.0101314, -0.0100249, -0.0101314, -0.0100249, -0.0000449, 0.0000449)
8: (-0.0042375, -0.0036610, -0.0042375, -0.0036610, -0.0002430, 0.0002430)
9: (0.0001568, 0.0030432, 0.0001568, 0.0030432, -0.0012166, 0.0012166)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.20 + 1.27 = 3.47 seconds
status: Status.ADV_EXAMPLE
