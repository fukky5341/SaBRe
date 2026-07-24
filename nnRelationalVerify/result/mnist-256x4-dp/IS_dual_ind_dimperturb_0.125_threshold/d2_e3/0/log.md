## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00379488


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0003269, 0.0003120, -0.0003269, 0.0003120, -0.0003892, 0.0003892)
1: (-0.0002101, 0.0017998, -0.0002101, 0.0017998, -0.0011901, 0.0011901)
2: (0.0136446, 0.0166547, 0.0136446, 0.0166547, -0.0016903, 0.0016903)
3: (-0.0003667, 0.0018967, -0.0003667, 0.0018967, -0.0012295, 0.0012295)
4: (-0.0047179, -0.0026301, -0.0047179, -0.0026301, -0.0014526, 0.0014526)
5: (0.0075721, 0.0098314, 0.0075721, 0.0098314, -0.0012233, 0.0012233)
6: (0.0091488, 0.0100754, 0.0091488, 0.0100754, -0.0009266, 0.0009266)
7: (-0.0197424, -0.0148377, -0.0197424, -0.0148377, -0.0022529, 0.0022529)
8: (0.9672266, 0.9812792, 0.9672266, 0.9812792, -0.0079849, 0.0079849)
9: (0.0031552, 0.0072853, 0.0031552, 0.0072853, -0.0020035, 0.0020035)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.38 + 1.31 = 2.70 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0055147, upper bound: 0.0055147

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 78

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.ADV_EXAMPLE
time: 0.43 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.ADV_EXAMPLE
time: 0.37 seconds

## IS Result
status: Status.ADV_EXAMPLE
execution time: (base) + (is) = 2.70 + 0.93 = 3.63 seconds
