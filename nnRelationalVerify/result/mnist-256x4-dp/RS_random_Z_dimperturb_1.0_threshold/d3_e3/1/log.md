## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 2.411746211


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716)
1: (-0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653)
2: (-0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282)
3: (-1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487)
4: (-1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829)
5: (-1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234)
6: (-1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365)
7: (-1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017)
8: (-1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066)
9: (-1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.46 + 3.83 = 5.29 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -2.4863377, upper bound: 2.4863377

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 175

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -2.3483483, upper bound: 2.3483483
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -2.3483483, upper bound: 2.3483483
time: 1.32 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 2.65 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 2.65
Output dim: 8, lower bound: -2.3483483, upper bound: 2.3483483
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 2.65
Output dim: 8, lower bound: -2.3483483, upper bound: 2.3483483

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 5.29 + 2.65 = 7.94 seconds
