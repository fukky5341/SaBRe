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
0: (-0.0041976, -0.0041849, -0.0041976, -0.0041849, -0.0000127, 0.0000127)
1: (-0.0098560, -0.0094819, -0.0098560, -0.0094819, -0.0003741, 0.0003741)
2: (0.9646359, 0.9650848, 0.9646359, 0.9650848, -0.0004489, 0.0004489)
3: (-0.0145337, -0.0112227, -0.0145337, -0.0112227, -0.0025047, 0.0025047)
4: (0.0001605, 0.0004123, 0.0001605, 0.0004123, -0.0002518, 0.0002518)
5: (0.0174326, 0.0177325, 0.0174326, 0.0177325, -0.0002999, 0.0002999)
6: (0.0031829, 0.0034276, 0.0031829, 0.0034276, -0.0002447, 0.0002447)
7: (-0.0048698, -0.0039399, -0.0048698, -0.0039399, -0.0009299, 0.0009299)
8: (0.0128657, 0.0135464, 0.0128657, 0.0135464, -0.0006807, 0.0006807)
9: (0.0208648, 0.0220892, 0.0208648, 0.0220892, -0.0011355, 0.0011355)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.25 + 1.35 = 2.61 seconds
status: Status.VERIFIED
relational distance
Output dim: 2, lower bound: -0.0002506, upper bound: 0.0002506
