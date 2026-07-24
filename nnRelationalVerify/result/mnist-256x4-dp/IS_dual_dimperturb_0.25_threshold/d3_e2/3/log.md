## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.06656274


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0159449, 0.0103625, -0.0159449, 0.0103625, -0.0263075, 0.0263075)
1: (-0.0249678, 0.0074213, -0.0249678, 0.0074213, -0.0323891, 0.0323891)
2: (0.0315386, 0.0615816, 0.0315386, 0.0615816, -0.0300430, 0.0300430)
3: (0.0004676, 0.0490425, 0.0004676, 0.0490425, -0.0421093, 0.0421093)
4: (-0.0189041, 0.0164137, -0.0189041, 0.0164137, -0.0353178, 0.0353178)
5: (-0.0015751, 0.0330188, -0.0015751, 0.0330188, -0.0345939, 0.0345939)
6: (-0.0439181, -0.0080551, -0.0439181, -0.0080551, -0.0358630, 0.0358630)
7: (0.8702401, 0.9669859, 0.8702401, 0.9669859, -0.0967458, 0.0967458)
8: (-0.0088935, 0.0369628, -0.0088935, 0.0369628, -0.0458564, 0.0458564)
9: (-0.0187208, 0.0130289, -0.0187208, 0.0130289, -0.0317497, 0.0317497)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.53 + 1.48 = 3.02 seconds
status: Status.VERIFIED
relational distance
Output dim: 7, lower bound: -0.0633732, upper bound: 0.0633732
