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
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0166500, 0.0175800, 0.0166500, 0.0175800, -0.0005725, 0.0005725)
1: (-0.0007911, -0.0001195, -0.0007911, -0.0001195, -0.0004226, 0.0004226)
2: (0.0037869, 0.0040879, 0.0037869, 0.0040879, -0.0001831, 0.0001831)
3: (0.0016580, 0.0022448, 0.0016580, 0.0022448, -0.0002988, 0.0002988)
4: (-0.0041519, -0.0034164, -0.0041519, -0.0034164, -0.0003649, 0.0003649)
5: (-0.0000797, 0.0003306, -0.0000797, 0.0003306, -0.0002595, 0.0002595)
6: (-0.0041126, -0.0026521, -0.0041126, -0.0026521, -0.0006634, 0.0006634)
7: (-0.0200676, -0.0158567, -0.0200676, -0.0158567, -0.0021105, 0.0021105)
8: (0.9769908, 0.9807326, 0.9769908, 0.9807326, -0.0019798, 0.0019798)
9: (0.0027176, 0.0054666, 0.0027176, 0.0054666, -0.0013931, 0.0013931)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.25 + 1.25 = 2.50 seconds
status: Status.ADV_EXAMPLE
