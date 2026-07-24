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
0: (-0.0041452, -0.0041304, -0.0041452, -0.0041304, -0.0000125, 0.0000125)
1: (-0.0078933, -0.0073389, -0.0078933, -0.0073389, -0.0004672, 0.0004672)
2: (0.9669911, 0.9676565, 0.9669911, 0.9676565, -0.0005606, 0.0005606)
3: (0.0028382, 0.0077457, 0.0028382, 0.0077457, -0.0041351, 0.0041351)
4: (-0.0012821, -0.0009089, -0.0012821, -0.0009089, -0.0003145, 0.0003145)
5: (0.0159745, 0.0163517, 0.0159745, 0.0163517, -0.0003179, 0.0003179)
6: (0.0039533, 0.0041368, 0.0039533, 0.0041368, -0.0001546, 0.0001546)
7: (-0.0097856, -0.0085138, -0.0097856, -0.0085138, -0.0010716, 0.0010716)
8: (0.0089657, 0.0099747, 0.0089657, 0.0099747, -0.0008502, 0.0008502)
9: (0.0138503, 0.0156651, 0.0138503, 0.0156651, -0.0015291, 0.0015291)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.47 + 1.24 = 2.71 seconds
status: Status.ADV_EXAMPLE
