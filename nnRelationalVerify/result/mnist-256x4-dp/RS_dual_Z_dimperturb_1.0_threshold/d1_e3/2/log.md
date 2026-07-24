## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0055854


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0044784, -0.0038492, -0.0044784, -0.0038492, -0.0005995, 0.0005995)
1: (0.0010607, 0.0045450, 0.0010607, 0.0045450, -0.0033195, 0.0033195)
2: (0.0048122, 0.0125964, 0.0048122, 0.0125964, -0.0074161, 0.0074161)
3: (0.0020262, 0.0053065, 0.0020262, 0.0053065, -0.0031252, 0.0031252)
4: (1.0046113, 1.0173376, 1.0046113, 1.0173376, -0.0121244, 0.0121244)
5: (0.0031385, 0.0056142, 0.0031385, 0.0056142, -0.0023587, 0.0023587)
6: (-0.0130490, -0.0098272, -0.0130490, -0.0098272, -0.0030695, 0.0030695)
7: (-0.0104679, -0.0100569, -0.0104679, -0.0100569, -0.0003915, 0.0003915)
8: (-0.0040643, -0.0018383, -0.0040643, -0.0018383, -0.0021208, 0.0021208)
9: (-0.0089681, 0.0021759, -0.0089681, 0.0021759, -0.0106170, 0.0106170)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.16 + 2.11 = 3.27 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -0.0062060, upper bound: 0.0062060

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0060313, upper bound: 0.0059506
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0059505, upper bound: 0.0060313
time: 1.25 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 2.60 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 2.60
Output dim: 4, lower bound: -0.0060313, upper bound: 0.0059506
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 2.60
Output dim: 4, lower bound: -0.0059505, upper bound: 0.0060313

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0044784, -0.0038492, -0.0044784, -0.0038492, -0.0005710, 0.0005733
1: 0.0010607, 0.0045450, 0.0010607, 0.0045450, -0.0031746, 0.0031614
2: 0.0048122, 0.0125964, 0.0048122, 0.0125964, -0.0070629, 0.0070924
3: 0.0020262, 0.0053065, 0.0020262, 0.0053065, -0.0029887, 0.0029763
4: 1.0046113, 1.0173376, 1.0046113, 1.0173376, -0.0115952, 0.0115471
5: 0.0031385, 0.0056142, 0.0031385, 0.0056142, -0.0022557, 0.0022464
6: -0.0130490, -0.0098272, -0.0130490, -0.0098272, -0.0029233, 0.0029355
7: -0.0104679, -0.0100569, -0.0104679, -0.0100569, -0.0003729, 0.0003744
8: -0.0040643, -0.0018383, -0.0040643, -0.0018383, -0.0020282, 0.0020198
9: -0.0089681, 0.0021759, -0.0089681, 0.0021759, -0.0101115, 0.0101536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0038109, upper bound: 0.0037978
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0038109, upper bound: 0.0037978
time: 0.76 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0044784, -0.0038492, -0.0044784, -0.0038492, -0.0005733, 0.0005710
1: 0.0010607, 0.0045450, 0.0010607, 0.0045450, -0.0031614, 0.0031746
2: 0.0048122, 0.0125964, 0.0048122, 0.0125964, -0.0070924, 0.0070629
3: 0.0020262, 0.0053065, 0.0020262, 0.0053065, -0.0029763, 0.0029887
4: 1.0046113, 1.0173376, 1.0046113, 1.0173376, -0.0115471, 0.0115952
5: 0.0031385, 0.0056142, 0.0031385, 0.0056142, -0.0022464, 0.0022557
6: -0.0130490, -0.0098272, -0.0130490, -0.0098272, -0.0029355, 0.0029233
7: -0.0104679, -0.0100569, -0.0104679, -0.0100569, -0.0003744, 0.0003729
8: -0.0040643, -0.0018383, -0.0040643, -0.0018383, -0.0020198, 0.0020282
9: -0.0089681, 0.0021759, -0.0089681, 0.0021759, -0.0101536, 0.0101115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0037978, upper bound: 0.0038109
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0037978, upper bound: 0.0038109
time: 0.78 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.73 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.73
Output dim: 4, lower bound: -0.0038109, upper bound: 0.0037978
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 2.73
Output dim: 4, lower bound: -0.0038109, upper bound: 0.0037978
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.73
Output dim: 4, lower bound: -0.0037978, upper bound: 0.0038109
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 2.73
Output dim: 4, lower bound: -0.0037978, upper bound: 0.0038109

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 3.27 + 8.01 = 11.28 seconds
