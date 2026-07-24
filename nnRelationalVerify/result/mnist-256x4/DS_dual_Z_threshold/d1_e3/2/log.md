## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0055854


## IAR start

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
execution time: IAR + RelationalAnalysis = 1.32 + 2.08 = 3.40 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -0.0062060, upper bound: 0.0062060

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0060313, upper bound: 0.0059506
time: 1.25 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0059505, upper bound: 0.0060313
time: 1.29 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 2.69 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 2.69
Output dim: 4, lower bound: -0.0060313, upper bound: 0.0059506
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 2.69
Output dim: 4, lower bound: -0.0059505, upper bound: 0.0060313

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0038109, upper bound: 0.0037978
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0038109, upper bound: 0.0037978
time: 0.74 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0037978, upper bound: 0.0038109
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0037978, upper bound: 0.0038109
time: 0.76 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 2.89 seconds
DS_DSZ1_DSZ1, status: Status.VERIFIED, split count: 2, time: 2.89
Output dim: 4, lower bound: -0.0038109, upper bound: 0.0037978
DS_DSZ1_DSZ2, status: Status.VERIFIED, split count: 2, time: 2.89
Output dim: 4, lower bound: -0.0038109, upper bound: 0.0037978
DS_DSZ2_DSZ1, status: Status.VERIFIED, split count: 2, time: 2.89
Output dim: 4, lower bound: -0.0037978, upper bound: 0.0038109
DS_DSZ2_DSZ2, status: Status.VERIFIED, split count: 2, time: 2.89
Output dim: 4, lower bound: -0.0037978, upper bound: 0.0038109

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 3.40 + 8.48 = 11.87 seconds
