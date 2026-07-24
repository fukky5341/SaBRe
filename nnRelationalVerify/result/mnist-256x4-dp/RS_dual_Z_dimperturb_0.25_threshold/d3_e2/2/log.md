## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00017725


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0041448, -0.0041300, -0.0041448, -0.0041300, -0.0000126, 0.0000126)
1: (-0.0078798, -0.0073236, -0.0078798, -0.0073236, -0.0004701, 0.0004701)
2: (0.9670073, 0.9676749, 0.9670073, 0.9676749, -0.0005641, 0.0005641)
3: (0.0029576, 0.0078812, 0.0029576, 0.0078812, -0.0041607, 0.0041607)
4: (-0.0012924, -0.0009180, -0.0012924, -0.0009180, -0.0003164, 0.0003164)
5: (0.0159641, 0.0163426, 0.0159641, 0.0163426, -0.0003198, 0.0003198)
6: (0.0039578, 0.0041418, 0.0039578, 0.0041418, -0.0001556, 0.0001556)
7: (-0.0098207, -0.0085447, -0.0098207, -0.0085447, -0.0010783, 0.0010783)
8: (0.0089378, 0.0099501, 0.0089378, 0.0099501, -0.0008555, 0.0008555)
9: (0.0138002, 0.0156209, 0.0138002, 0.0156209, -0.0015386, 0.0015386)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.61 + 1.26 = 2.88 seconds
status: Status.ADV_EXAMPLE
