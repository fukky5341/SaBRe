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
0: (-0.0016807, -0.0015308, -0.0016807, -0.0015308, -0.0000584, 0.0000584)
1: (-0.0085757, -0.0081951, -0.0085757, -0.0081951, -0.0001481, 0.0001481)
2: (0.0297096, 0.0299458, 0.0297096, 0.0299458, -0.0000919, 0.0000919)
3: (0.0036462, 0.0040871, 0.0036462, 0.0040871, -0.0001716, 0.0001716)
4: (-0.0076159, -0.0072288, -0.0076159, -0.0072288, -0.0001507, 0.0001507)
5: (0.0108535, 0.0110001, 0.0108535, 0.0110001, -0.0000571, 0.0000571)
6: (0.0050156, 0.0055752, 0.0050156, 0.0055752, -0.0002178, 0.0002178)
7: (0.9815689, 0.9819605, 0.9815689, 0.9819605, -0.0001524, 0.0001524)
8: (-0.0063252, -0.0059054, -0.0063252, -0.0059054, -0.0001634, 0.0001634)
9: (-0.0010987, -0.0008214, -0.0010987, -0.0008214, -0.0001079, 0.0001079)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.55 + 1.19 = 2.74 seconds
status: Status.ADV_EXAMPLE
