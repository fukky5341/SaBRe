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
Threshold: 0.00039634


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0170518, 0.0174643, 0.0170518, 0.0174643, -0.0002275, 0.0002275)
1: (-0.0004931, -0.0002120, -0.0004931, -0.0002120, -0.0001658, 0.0001658)
2: (0.0038241, 0.0039597, 0.0038241, 0.0039597, -0.0000732, 0.0000732)
3: (0.0017626, 0.0020596, 0.0017626, 0.0020596, -0.0001512, 0.0001512)
4: (-0.0040364, -0.0036341, -0.0040364, -0.0036341, -0.0001751, 0.0001751)
5: (-0.0000198, 0.0001477, -0.0000198, 0.0001477, -0.0001015, 0.0001015)
6: (-0.0037964, -0.0029965, -0.0037964, -0.0029965, -0.0003609, 0.0003609)
7: (-0.0194202, -0.0171341, -0.0194202, -0.0171341, -0.0009977, 0.0009977)
8: (0.9775149, 0.9794550, 0.9775149, 0.9794550, -0.0008720, 0.0008720)
9: (0.0035737, 0.0050526, 0.0035737, 0.0050526, -0.0006482, 0.0006482)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.33 + 1.15 = 2.49 seconds
status: Status.ADV_EXAMPLE
