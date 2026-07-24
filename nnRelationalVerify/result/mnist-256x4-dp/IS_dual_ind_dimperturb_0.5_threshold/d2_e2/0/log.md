## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00046428


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0041770, -0.0041475, -0.0041770, -0.0041475, -0.0000274, 0.0000274)
1: (-0.0090828, -0.0079793, -0.0090828, -0.0079793, -0.0010276, 0.0010276)
2: (0.9655637, 0.9668880, 0.9655637, 0.9668880, -0.0012331, 0.0012331)
3: (-0.0076906, 0.0020767, -0.0076906, 0.0020767, -0.0090955, 0.0090955)
4: (-0.0008510, -0.0001081, -0.0008510, -0.0001081, -0.0006918, 0.0006918)
5: (0.0164103, 0.0171611, 0.0164103, 0.0171611, -0.0006991, 0.0006991)
6: (0.0035596, 0.0039248, 0.0035596, 0.0039248, -0.0003401, 0.0003401)
7: (-0.0083164, -0.0057852, -0.0083164, -0.0057852, -0.0023572, 0.0023572)
8: (0.0101313, 0.0121395, 0.0101313, 0.0121395, -0.0018701, 0.0018701)
9: (0.0159467, 0.0195586, 0.0159467, 0.0195586, -0.0033635, 0.0033635)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.21 + 1.33 = 2.54 seconds
status: Status.ADV_EXAMPLE
