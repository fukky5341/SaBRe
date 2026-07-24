## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00104286


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0041979, -0.0041809, -0.0041979, -0.0041809, -0.0000170, 0.0000170)
1: (-0.0098678, -0.0094847, -0.0098678, -0.0094847, -0.0003831, 0.0003831)
2: (0.9646217, 0.9650814, 0.9646217, 0.9650814, -0.0004597, 0.0004597)
3: (-0.0146390, -0.0112478, -0.0146390, -0.0112478, -0.0025413, 0.0025413)
4: (0.0001624, 0.0004204, 0.0001624, 0.0004204, -0.0002579, 0.0002579)
5: (0.0174345, 0.0177560, 0.0174345, 0.0177560, -0.0003215, 0.0003215)
6: (0.0031378, 0.0034266, 0.0031378, 0.0034266, -0.0002888, 0.0002888)
7: (-0.0048633, -0.0038882, -0.0048633, -0.0038882, -0.0009751, 0.0009751)
8: (0.0128708, 0.0135681, 0.0128708, 0.0135681, -0.0006972, 0.0006972)
9: (0.0208740, 0.0221281, 0.0208740, 0.0221281, -0.0011607, 0.0011607)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.18 + 1.36 = 2.54 seconds
status: Status.VERIFIED
relational distance
Output dim: 2, lower bound: -0.0002595, upper bound: 0.0002595
