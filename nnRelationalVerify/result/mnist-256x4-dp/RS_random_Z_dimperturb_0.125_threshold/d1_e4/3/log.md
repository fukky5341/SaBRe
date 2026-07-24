## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 8.44e-05


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0016860, -0.0015065, -0.0016860, -0.0015065, -0.0000678, 0.0000678)
1: (-0.0085891, -0.0081336, -0.0085891, -0.0081336, -0.0001721, 0.0001721)
2: (0.0297013, 0.0299839, 0.0297013, 0.0299839, -0.0001068, 0.0001068)
3: (0.0035749, 0.0041027, 0.0035749, 0.0041027, -0.0001994, 0.0001994)
4: (-0.0076296, -0.0071662, -0.0076296, -0.0071662, -0.0001751, 0.0001751)
5: (0.0108483, 0.0110238, 0.0108483, 0.0110238, -0.0000663, 0.0000663)
6: (0.0049252, 0.0055950, 0.0049252, 0.0055950, -0.0002531, 0.0002531)
7: (0.9815057, 0.9819744, 0.9815057, 0.9819744, -0.0001771, 0.0001771)
8: (-0.0063931, -0.0058906, -0.0063931, -0.0058906, -0.0001899, 0.0001899)
9: (-0.0011086, -0.0007766, -0.0011086, -0.0007766, -0.0001254, 0.0001254)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.46 + 1.20 = 2.66 seconds
status: Status.ADV_EXAMPLE
