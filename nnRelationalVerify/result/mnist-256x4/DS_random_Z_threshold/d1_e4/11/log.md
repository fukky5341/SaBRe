## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 11)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.014896192999999999


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959)
1: (0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687)
2: (-0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0327574, 0.0327574)
3: (-0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814)
4: (-0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126)
5: (-0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610)
6: (-0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640)
7: (-0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709)
8: (-0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0398314, 0.0398314)
9: (-0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.18 + 3.67 = 4.85 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.0153569, upper bound: 0.0153569

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0153324, upper bound: 0.0153323
time: 2.62 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0153324, upper bound: 0.0153324
time: 2.59 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 5.22 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 5.22
Output dim: 1, lower bound: -0.0153324, upper bound: 0.0153323
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 5.22
Output dim: 1, lower bound: -0.0153324, upper bound: 0.0153324

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0327452, 0.0327466
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0398286, 0.0398282
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 221

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0153314, upper bound: 0.0153324
time: 2.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0153314, upper bound: 0.0153314
time: 2.63 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0327466, 0.0327452
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0398282, 0.0398286
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 130

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 52

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0153201, upper bound: 0.0153210
time: 2.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0153201, upper bound: 0.0153201
time: 2.55 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 6.16 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 6.16
Output dim: 1, lower bound: -0.0153314, upper bound: 0.0153324
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 6.16
Output dim: 1, lower bound: -0.0153314, upper bound: 0.0153314
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 6.16
Output dim: 1, lower bound: -0.0153201, upper bound: 0.0153210
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 6.16
Output dim: 1, lower bound: -0.0153201, upper bound: 0.0153201

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0327440, 0.0327486
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0398291, 0.0398279
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 230

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150871, upper bound: 0.0150880
time: 2.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150871, upper bound: 0.0150880
time: 2.61 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0327472, 0.0327453
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0398282, 0.0398287
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 134

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0153200, upper bound: 0.0153205
time: 2.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0153200, upper bound: 0.0153200
time: 2.66 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0327449, 0.0327440
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0398279, 0.0398282
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 131

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0152437, upper bound: 0.0152566
time: 2.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0152437, upper bound: 0.0152440
time: 2.71 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0327454, 0.0327435
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0398278, 0.0398283
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 229

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0152862, upper bound: 0.0153022
time: 2.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0153033, upper bound: 0.0152862
time: 2.85 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 6.45 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 6.45
Output dim: 1, lower bound: -0.0150871, upper bound: 0.0150880
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 6.45
Output dim: 1, lower bound: -0.0150871, upper bound: 0.0150880
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 6.45
Output dim: 1, lower bound: -0.0153200, upper bound: 0.0153205
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 6.45
Output dim: 1, lower bound: -0.0153200, upper bound: 0.0153200
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 6.45
Output dim: 1, lower bound: -0.0152437, upper bound: 0.0152566
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 6.45
Output dim: 1, lower bound: -0.0152437, upper bound: 0.0152440
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 6.45
Output dim: 1, lower bound: -0.0152862, upper bound: 0.0153022
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 6.45
Output dim: 1, lower bound: -0.0153033, upper bound: 0.0152862

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0327185, 0.0327259
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0398232, 0.0398212
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 173

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0148965, upper bound: 0.0148968
time: 2.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0148965, upper bound: 0.0148968
time: 2.34 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0327213, 0.0327220
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0398221, 0.0398220
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150229, upper bound: 0.0150244
time: 2.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150229, upper bound: 0.0150241
time: 2.72 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0327310, 0.0327294
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0398254, 0.0398259
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149400, upper bound: 0.0149400
time: 1.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149400, upper bound: 0.0149400
time: 1.85 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0327312, 0.0327281
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0398251, 0.0398259
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 230

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150750, upper bound: 0.0150750
time: 2.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150750, upper bound: 0.0150750
time: 2.67 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0327372, 0.0327377
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0398263, 0.0398262
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 88

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0152133, upper bound: 0.0152342
time: 2.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0152133, upper bound: 0.0152271
time: 2.55 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0327386, 0.0327362
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0398259, 0.0398265
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151323, upper bound: 0.0151323
time: 2.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151323, upper bound: 0.0151323
time: 2.58 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0325885, 0.0326081
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397917, 0.0397864
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 196

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0152186, upper bound: 0.0152349
time: 2.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0152186, upper bound: 0.0152348
time: 2.70 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0326101, 0.0325866
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397859, 0.0397922
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 88

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 108

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0153004, upper bound: 0.0152862
time: 2.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0153033, upper bound: 0.0152862
time: 3.09 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 6.87 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 6.87
Output dim: 1, lower bound: -0.0148965, upper bound: 0.0148968
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 6.87
Output dim: 1, lower bound: -0.0148965, upper bound: 0.0148968
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 6.87
Output dim: 1, lower bound: -0.0150229, upper bound: 0.0150244
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 6.87
Output dim: 1, lower bound: -0.0150229, upper bound: 0.0150241
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 6.87
Output dim: 1, lower bound: -0.0149400, upper bound: 0.0149400
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 6.87
Output dim: 1, lower bound: -0.0149400, upper bound: 0.0149400
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 6.87
Output dim: 1, lower bound: -0.0150750, upper bound: 0.0150750
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 6.87
Output dim: 1, lower bound: -0.0150750, upper bound: 0.0150750
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 6.87
Output dim: 1, lower bound: -0.0152133, upper bound: 0.0152342
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 6.87
Output dim: 1, lower bound: -0.0152133, upper bound: 0.0152271
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 6.87
Output dim: 1, lower bound: -0.0151323, upper bound: 0.0151323
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 6.87
Output dim: 1, lower bound: -0.0151323, upper bound: 0.0151323
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 6.87
Output dim: 1, lower bound: -0.0152186, upper bound: 0.0152349
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 6.87
Output dim: 1, lower bound: -0.0152186, upper bound: 0.0152348
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 6.87
Output dim: 1, lower bound: -0.0153004, upper bound: 0.0152862
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 6.87
Output dim: 1, lower bound: -0.0153033, upper bound: 0.0152862

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0326809, 0.0326923
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0398140, 0.0398109
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 88

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 203

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0141611, upper bound: 0.0141610
time: 1.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0141611, upper bound: 0.0141610
time: 1.89 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0326848, 0.0326878
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0398128, 0.0398120
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 203

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0141611, upper bound: 0.0141610
time: 1.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0141611, upper bound: 0.0141610
time: 1.88 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0326761, 0.0326676
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0398070, 0.0398093
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 229

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150072, upper bound: 0.0150086
time: 2.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150072, upper bound: 0.0150086
time: 2.72 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0326669, 0.0326769
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0398095, 0.0398069
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 131

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0148405, upper bound: 0.0148414
time: 2.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0148405, upper bound: 0.0148414
time: 2.39 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0327230, 0.0327207
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0398232, 0.0398238
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 229

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149090, upper bound: 0.0149212
time: 2.05 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149209, upper bound: 0.0149098
time: 2.06 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0327223, 0.0327294
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0398254, 0.0398236
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 131

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 230

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0146335, upper bound: 0.0146337
time: 1.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0146335, upper bound: 0.0146337
time: 1.51 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0327046, 0.0327052
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0398192, 0.0398190
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 198

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150399, upper bound: 0.0150453
time: 2.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150399, upper bound: 0.0150399
time: 2.71 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0327084, 0.0327031
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0398186, 0.0398200
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149828, upper bound: 0.0149888
time: 2.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149819, upper bound: 0.0149819
time: 2.89 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0326207, 0.0326383
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397999, 0.0397952
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 212

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151703, upper bound: 0.0151916
time: 2.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151703, upper bound: 0.0151883
time: 2.74 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0326377, 0.0326212
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397953, 0.0397997
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 134

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0152013, upper bound: 0.0152163
time: 2.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0152013, upper bound: 0.0152151
time: 2.84 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0327327, 0.0327311
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0398245, 0.0398249
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 230

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0147957, upper bound: 0.0147908
time: 33.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0147957, upper bound: 0.0147908
time: 39.32 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0327386, 0.0327304
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0398242, 0.0398265
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151273, upper bound: 0.0151276
time: 2.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151273, upper bound: 0.0151289
time: 2.73 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0325415, 0.0325529
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397765, 0.0397735
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 203

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149474, upper bound: 0.0149581
time: 2.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149474, upper bound: 0.0149581
time: 2.47 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0325332, 0.0325613
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397788, 0.0397713
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151103, upper bound: 0.0151210
time: 2.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151103, upper bound: 0.0151210
time: 2.67 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0326034, 0.0325805
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397843, 0.0397904
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 107

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149471, upper bound: 0.0149350
time: 2.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149471, upper bound: 0.0149350
time: 2.26 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0326042, 0.0325798
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397841, 0.0397906
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 130

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0148446, upper bound: 0.0148378
time: 2.09 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0148446, upper bound: 0.0148378
time: 2.09 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 5.32 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 5.32
Output dim: 1, lower bound: -0.0141611, upper bound: 0.0141610
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 5.32
Output dim: 1, lower bound: -0.0141611, upper bound: 0.0141610
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 5.32
Output dim: 1, lower bound: -0.0141611, upper bound: 0.0141610
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 5.32
Output dim: 1, lower bound: -0.0141611, upper bound: 0.0141610
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.32
Output dim: 1, lower bound: -0.0150072, upper bound: 0.0150086
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.32
Output dim: 1, lower bound: -0.0150072, upper bound: 0.0150086
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 5.32
Output dim: 1, lower bound: -0.0148405, upper bound: 0.0148414
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 5.32
Output dim: 1, lower bound: -0.0148405, upper bound: 0.0148414
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.32
Output dim: 1, lower bound: -0.0149090, upper bound: 0.0149212
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.32
Output dim: 1, lower bound: -0.0149209, upper bound: 0.0149098
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 5.32
Output dim: 1, lower bound: -0.0146335, upper bound: 0.0146337
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 5.32
Output dim: 1, lower bound: -0.0146335, upper bound: 0.0146337
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.32
Output dim: 1, lower bound: -0.0150399, upper bound: 0.0150453
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.32
Output dim: 1, lower bound: -0.0150399, upper bound: 0.0150399
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.32
Output dim: 1, lower bound: -0.0149828, upper bound: 0.0149888
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.32
Output dim: 1, lower bound: -0.0149819, upper bound: 0.0149819
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.32
Output dim: 1, lower bound: -0.0151703, upper bound: 0.0151916
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.32
Output dim: 1, lower bound: -0.0151703, upper bound: 0.0151883
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.32
Output dim: 1, lower bound: -0.0152013, upper bound: 0.0152163
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.32
Output dim: 1, lower bound: -0.0152013, upper bound: 0.0152151
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 5.32
Output dim: 1, lower bound: -0.0147957, upper bound: 0.0147908
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 5.32
Output dim: 1, lower bound: -0.0147957, upper bound: 0.0147908
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.32
Output dim: 1, lower bound: -0.0151273, upper bound: 0.0151276
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.32
Output dim: 1, lower bound: -0.0151273, upper bound: 0.0151289
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.32
Output dim: 1, lower bound: -0.0149474, upper bound: 0.0149581
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.32
Output dim: 1, lower bound: -0.0149474, upper bound: 0.0149581
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.32
Output dim: 1, lower bound: -0.0151103, upper bound: 0.0151210
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.32
Output dim: 1, lower bound: -0.0151103, upper bound: 0.0151210
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.32
Output dim: 1, lower bound: -0.0149471, upper bound: 0.0149350
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.32
Output dim: 1, lower bound: -0.0149471, upper bound: 0.0149350
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 5.32
Output dim: 1, lower bound: -0.0148446, upper bound: 0.0148378
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 5.32
Output dim: 1, lower bound: -0.0148446, upper bound: 0.0148378

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0326354, 0.0326258
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397964, 0.0397989
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 107

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0146219, upper bound: 0.0146221
time: 1.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0146219, upper bound: 0.0146221
time: 1.84 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0326343, 0.0326253
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397962, 0.0397986
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0145546, upper bound: 0.0145553
time: 2.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0145546, upper bound: 0.0145553
time: 2.02 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0325687, 0.0325883
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397876, 0.0397824
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 90

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0148944, upper bound: 0.0149042
time: 2.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0148925, upper bound: 0.0149066
time: 2.13 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0325900, 0.0325664
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397818, 0.0397881
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 230

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 203

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 173

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0147300, upper bound: 0.0147186
time: 1.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0147300, upper bound: 0.0147186
time: 1.60 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0326963, 0.0326981
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0398173, 0.0398168
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 212

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149918, upper bound: 0.0150010
time: 5.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149918, upper bound: 0.0149974
time: 2.47 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0326975, 0.0326971
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0398170, 0.0398171
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0147167, upper bound: 0.0147131
time: 1.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0147167, upper bound: 0.0147131
time: 1.75 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0327005, 0.0326967
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0398170, 0.0398180
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0145745, upper bound: 0.0145811
time: 1.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0145745, upper bound: 0.0145811
time: 1.71 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0327019, 0.0326954
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0398166, 0.0398184
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 108

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 107

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0145960, upper bound: 0.0145922
time: 2.01 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0145960, upper bound: 0.0145922
time: 2.03 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0326206, 0.0326382
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397999, 0.0397952
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0148323, upper bound: 0.0148389
time: 1.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0148323, upper bound: 0.0148389
time: 1.90 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0326206, 0.0326381
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397999, 0.0397952
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 203

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0147177, upper bound: 0.0147288
time: 2.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0147177, upper bound: 0.0147288
time: 2.19 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0326221, 0.0326056
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397925, 0.0397970
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 130

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 85

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0146972, upper bound: 0.0147021
time: 1.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0146972, upper bound: 0.0147021
time: 1.87 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0326221, 0.0326053
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397925, 0.0397970
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 196

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150918, upper bound: 0.0151052
time: 2.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150918, upper bound: 0.0151052
time: 2.31 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0327325, 0.0327276
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0398232, 0.0398246
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 229

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151113, upper bound: 0.0151090
time: 2.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150976, upper bound: 0.0150979
time: 2.42 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0327358, 0.0327242
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0398223, 0.0398255
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 195

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150841, upper bound: 0.0150857
time: 2.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150841, upper bound: 0.0150857
time: 2.24 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0325364, 0.0325426
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397737, 0.0397721
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 130

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0141756, upper bound: 0.0141776
time: 1.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0141756, upper bound: 0.0141776
time: 1.29 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0325313, 0.0325529
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397765, 0.0397707
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 130

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 131

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 251

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0148956, upper bound: 0.0149061
time: 1.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0148957, upper bound: 0.0149061
time: 2.02 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0325273, 0.0325561
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397773, 0.0397696
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 88

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150770, upper bound: 0.0150972
time: 2.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150851, upper bound: 0.0150920
time: 2.72 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0325332, 0.0325554
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397772, 0.0397713
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 203

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150938, upper bound: 0.0151051
time: 2.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150938, upper bound: 0.0151061
time: 2.46 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0325997, 0.0325780
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397836, 0.0397894
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0145199, upper bound: 0.0145175
time: 1.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0145199, upper bound: 0.0145175
time: 1.44 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0326009, 0.0325805
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397843, 0.0397897
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 203

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 198

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149058, upper bound: 0.0149094
time: 2.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149058, upper bound: 0.0149058
time: 2.12 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 5.46 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 5.46
Output dim: 1, lower bound: -0.0146219, upper bound: 0.0146221
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 5.46
Output dim: 1, lower bound: -0.0146219, upper bound: 0.0146221
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 5.46
Output dim: 1, lower bound: -0.0145546, upper bound: 0.0145553
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 5.46
Output dim: 1, lower bound: -0.0145546, upper bound: 0.0145553
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.46
Output dim: 1, lower bound: -0.0148944, upper bound: 0.0149042
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.46
Output dim: 1, lower bound: -0.0148925, upper bound: 0.0149066
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 5.46
Output dim: 1, lower bound: -0.0147300, upper bound: 0.0147186
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 5.46
Output dim: 1, lower bound: -0.0147300, upper bound: 0.0147186
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.46
Output dim: 1, lower bound: -0.0149918, upper bound: 0.0150010
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.46
Output dim: 1, lower bound: -0.0149918, upper bound: 0.0149974
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 5.46
Output dim: 1, lower bound: -0.0147167, upper bound: 0.0147131
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 5.46
Output dim: 1, lower bound: -0.0147167, upper bound: 0.0147131
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 5.46
Output dim: 1, lower bound: -0.0145745, upper bound: 0.0145811
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 5.46
Output dim: 1, lower bound: -0.0145745, upper bound: 0.0145811
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 5.46
Output dim: 1, lower bound: -0.0145960, upper bound: 0.0145922
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 5.46
Output dim: 1, lower bound: -0.0145960, upper bound: 0.0145922
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 5.46
Output dim: 1, lower bound: -0.0148323, upper bound: 0.0148389
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 5.46
Output dim: 1, lower bound: -0.0148323, upper bound: 0.0148389
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 5.46
Output dim: 1, lower bound: -0.0147177, upper bound: 0.0147288
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 5.46
Output dim: 1, lower bound: -0.0147177, upper bound: 0.0147288
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 5.46
Output dim: 1, lower bound: -0.0146972, upper bound: 0.0147021
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 5.46
Output dim: 1, lower bound: -0.0146972, upper bound: 0.0147021
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.46
Output dim: 1, lower bound: -0.0150918, upper bound: 0.0151052
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.46
Output dim: 1, lower bound: -0.0150918, upper bound: 0.0151052
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.46
Output dim: 1, lower bound: -0.0151113, upper bound: 0.0151090
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.46
Output dim: 1, lower bound: -0.0150976, upper bound: 0.0150979
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.46
Output dim: 1, lower bound: -0.0150841, upper bound: 0.0150857
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.46
Output dim: 1, lower bound: -0.0150841, upper bound: 0.0150857
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 5.46
Output dim: 1, lower bound: -0.0141756, upper bound: 0.0141776
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 5.46
Output dim: 1, lower bound: -0.0141756, upper bound: 0.0141776
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.46
Output dim: 1, lower bound: -0.0148956, upper bound: 0.0149061
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.46
Output dim: 1, lower bound: -0.0148957, upper bound: 0.0149061
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.46
Output dim: 1, lower bound: -0.0150770, upper bound: 0.0150972
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.46
Output dim: 1, lower bound: -0.0150851, upper bound: 0.0150920
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.46
Output dim: 1, lower bound: -0.0150938, upper bound: 0.0151051
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.46
Output dim: 1, lower bound: -0.0150938, upper bound: 0.0151061
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 5.46
Output dim: 1, lower bound: -0.0145199, upper bound: 0.0145175
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 5.46
Output dim: 1, lower bound: -0.0145199, upper bound: 0.0145175
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.46
Output dim: 1, lower bound: -0.0149058, upper bound: 0.0149094
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.46
Output dim: 1, lower bound: -0.0149058, upper bound: 0.0149058

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0325251, 0.0325459
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397770, 0.0397715
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 173

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 107

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0143814, upper bound: 0.0143840
time: 1.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0143814, upper bound: 0.0143840
time: 1.41 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0325263, 0.0325466
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397772, 0.0397718
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 198

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0148595, upper bound: 0.0148747
time: 2.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0148582, upper bound: 0.0148705
time: 2.44 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0326962, 0.0326980
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0398172, 0.0398168
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 173

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0145546, upper bound: 0.0145581
time: 1.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0145546, upper bound: 0.0145581
time: 1.94 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0326962, 0.0326980
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0398172, 0.0398168
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 203

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 90

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0145621, upper bound: 0.0145592
time: 2.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0145621, upper bound: 0.0145592
time: 2.06 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0326161, 0.0326002
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397911, 0.0397953
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 173

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 108

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150959, upper bound: 0.0151052
time: 2.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150959, upper bound: 0.0151045
time: 2.51 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0326221, 0.0325993
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397908, 0.0397970
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 173

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 229

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150597, upper bound: 0.0150876
time: 2.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150664, upper bound: 0.0150732
time: 2.57 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0325749, 0.0325912
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397868, 0.0397825
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 221

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150975, upper bound: 0.0151090
time: 2.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150975, upper bound: 0.0151085
time: 2.37 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0325967, 0.0325700
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397811, 0.0397883
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151086, upper bound: 0.0150817
time: 2.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150805, upper bound: 0.0150840
time: 2.40 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0327161, 0.0327082
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0398180, 0.0398202
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 230

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0147777, upper bound: 0.0147736
time: 2.07 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0147777, upper bound: 0.0147736
time: 2.02 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0327199, 0.0327045
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0398170, 0.0398212
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 131

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149107, upper bound: 0.0149025
time: 2.05 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149107, upper bound: 0.0149025
time: 2.05 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0324407, 0.0324596
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397531, 0.0397480
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 173

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 195

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0148634, upper bound: 0.0148722
time: 2.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0148634, upper bound: 0.0148722
time: 2.28 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0324380, 0.0324598
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397531, 0.0397473
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 196

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 212

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0148390, upper bound: 0.0148548
time: 2.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0148390, upper bound: 0.0148505
time: 2.29 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0324091, 0.0324532
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397505, 0.0397387
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 130

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150725, upper bound: 0.0150922
time: 3.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150725, upper bound: 0.0150938
time: 2.63 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0324260, 0.0324378
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397464, 0.0397432
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150725, upper bound: 0.0150873
time: 2.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150725, upper bound: 0.0150886
time: 2.31 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0324908, 0.0325145
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397666, 0.0397603
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 173

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0148739, upper bound: 0.0148845
time: 2.08 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0148739, upper bound: 0.0148845
time: 2.37 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0324922, 0.0325152
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397667, 0.0397607
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 131

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 173

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0148729, upper bound: 0.0148855
time: 2.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0148729, upper bound: 0.0148855
time: 2.33 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0325926, 0.0325734
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397824, 0.0397875
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0148532, upper bound: 0.0148569
time: 2.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0148532, upper bound: 0.0148566
time: 2.51 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0325938, 0.0325723
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397821, 0.0397878
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0148698, upper bound: 0.0148721
time: 2.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0148893, upper bound: 0.0148697
time: 2.27 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 5.93 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.93
Output dim: 1, lower bound: -0.0143814, upper bound: 0.0143840
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.93
Output dim: 1, lower bound: -0.0143814, upper bound: 0.0143840
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.93
Output dim: 1, lower bound: -0.0148595, upper bound: 0.0148747
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.93
Output dim: 1, lower bound: -0.0148582, upper bound: 0.0148705
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.93
Output dim: 1, lower bound: -0.0145546, upper bound: 0.0145581
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.93
Output dim: 1, lower bound: -0.0145546, upper bound: 0.0145581
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.93
Output dim: 1, lower bound: -0.0145621, upper bound: 0.0145592
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.93
Output dim: 1, lower bound: -0.0145621, upper bound: 0.0145592
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.93
Output dim: 1, lower bound: -0.0150959, upper bound: 0.0151052
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.93
Output dim: 1, lower bound: -0.0150959, upper bound: 0.0151045
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.93
Output dim: 1, lower bound: -0.0150597, upper bound: 0.0150876
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.93
Output dim: 1, lower bound: -0.0150664, upper bound: 0.0150732
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.93
Output dim: 1, lower bound: -0.0150975, upper bound: 0.0151090
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.93
Output dim: 1, lower bound: -0.0150975, upper bound: 0.0151085
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.93
Output dim: 1, lower bound: -0.0151086, upper bound: 0.0150817
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.93
Output dim: 1, lower bound: -0.0150805, upper bound: 0.0150840
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.93
Output dim: 1, lower bound: -0.0147777, upper bound: 0.0147736
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.93
Output dim: 1, lower bound: -0.0147777, upper bound: 0.0147736
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.93
Output dim: 1, lower bound: -0.0149107, upper bound: 0.0149025
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.93
Output dim: 1, lower bound: -0.0149107, upper bound: 0.0149025
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.93
Output dim: 1, lower bound: -0.0148634, upper bound: 0.0148722
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.93
Output dim: 1, lower bound: -0.0148634, upper bound: 0.0148722
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.93
Output dim: 1, lower bound: -0.0148390, upper bound: 0.0148548
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.93
Output dim: 1, lower bound: -0.0148390, upper bound: 0.0148505
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.93
Output dim: 1, lower bound: -0.0150725, upper bound: 0.0150922
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.93
Output dim: 1, lower bound: -0.0150725, upper bound: 0.0150938
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.93
Output dim: 1, lower bound: -0.0150725, upper bound: 0.0150873
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.93
Output dim: 1, lower bound: -0.0150725, upper bound: 0.0150886
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.93
Output dim: 1, lower bound: -0.0148739, upper bound: 0.0148845
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.93
Output dim: 1, lower bound: -0.0148739, upper bound: 0.0148845
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.93
Output dim: 1, lower bound: -0.0148729, upper bound: 0.0148855
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.93
Output dim: 1, lower bound: -0.0148729, upper bound: 0.0148855
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.93
Output dim: 1, lower bound: -0.0148532, upper bound: 0.0148569
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.93
Output dim: 1, lower bound: -0.0148532, upper bound: 0.0148566
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.93
Output dim: 1, lower bound: -0.0148698, upper bound: 0.0148721
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.93
Output dim: 1, lower bound: -0.0148893, upper bound: 0.0148697

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0326092, 0.0325937
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397894, 0.0397935
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150689, upper bound: 0.0150790
time: 2.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150699, upper bound: 0.0150779
time: 2.78 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0326097, 0.0325933
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397893, 0.0397936
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 251

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150583, upper bound: 0.0150660
time: 2.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150598, upper bound: 0.0150660
time: 2.58 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0324667, 0.0324666
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397554, 0.0397555
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 221

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150590, upper bound: 0.0150876
time: 2.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150590, upper bound: 0.0150859
time: 2.56 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0324868, 0.0324441
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397494, 0.0397609
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 131

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0148654, upper bound: 0.0148704
time: 2.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0148654, upper bound: 0.0148704
time: 2.56 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0325736, 0.0325931
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397873, 0.0397822
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150804, upper bound: 0.0150930
time: 2.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150938, upper bound: 0.0150945
time: 2.32 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0325768, 0.0325904
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397866, 0.0397831
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 107

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0147343, upper bound: 0.0147327
time: 1.90 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0147343, upper bound: 0.0147327
time: 1.88 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0325553, 0.0325290
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397705, 0.0397776
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 88

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150143, upper bound: 0.0150152
time: 2.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150143, upper bound: 0.0150152
time: 2.69 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0325556, 0.0325279
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397702, 0.0397777
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 203

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150522, upper bound: 0.0150574
time: 2.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150522, upper bound: 0.0150553
time: 3.01 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0327169, 0.0327025
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0398165, 0.0398204
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0148369, upper bound: 0.0148287
time: 2.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0148271, upper bound: 0.0148284
time: 2.21 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0327180, 0.0327045
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0398170, 0.0398207
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 108

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 198

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0148675, upper bound: 0.0148686
time: 2.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0148757, upper bound: 0.0148683
time: 2.10 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0324023, 0.0324492
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397491, 0.0397366
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 221

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150714, upper bound: 0.0150922
time: 2.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150714, upper bound: 0.0150914
time: 2.51 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0324057, 0.0324464
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397483, 0.0397375
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 131

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150455, upper bound: 0.0150669
time: 2.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150455, upper bound: 0.0150659
time: 2.68 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0324192, 0.0324341
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397451, 0.0397411
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 107

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0147266, upper bound: 0.0147268
time: 2.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0147266, upper bound: 0.0147268
time: 2.21 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0324223, 0.0324310
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397443, 0.0397419
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 195

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150474, upper bound: 0.0150523
time: 2.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150474, upper bound: 0.0150523
time: 2.63 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 8.31 seconds
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 8.31
Output dim: 1, lower bound: -0.0150689, upper bound: 0.0150790
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 8.31
Output dim: 1, lower bound: -0.0150699, upper bound: 0.0150779
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 8.31
Output dim: 1, lower bound: -0.0150583, upper bound: 0.0150660
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 8.31
Output dim: 1, lower bound: -0.0150598, upper bound: 0.0150660
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 8.31
Output dim: 1, lower bound: -0.0150590, upper bound: 0.0150876
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 8.31
Output dim: 1, lower bound: -0.0150590, upper bound: 0.0150859
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 8.31
Output dim: 1, lower bound: -0.0148654, upper bound: 0.0148704
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 8.31
Output dim: 1, lower bound: -0.0148654, upper bound: 0.0148704
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 8.31
Output dim: 1, lower bound: -0.0150804, upper bound: 0.0150930
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 8.31
Output dim: 1, lower bound: -0.0150938, upper bound: 0.0150945
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 8.31
Output dim: 1, lower bound: -0.0147343, upper bound: 0.0147327
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 8.31
Output dim: 1, lower bound: -0.0147343, upper bound: 0.0147327
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 8.31
Output dim: 1, lower bound: -0.0150143, upper bound: 0.0150152
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 8.31
Output dim: 1, lower bound: -0.0150143, upper bound: 0.0150152
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 8.31
Output dim: 1, lower bound: -0.0150522, upper bound: 0.0150574
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 8.31
Output dim: 1, lower bound: -0.0150522, upper bound: 0.0150553
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 8.31
Output dim: 1, lower bound: -0.0148369, upper bound: 0.0148287
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 8.31
Output dim: 1, lower bound: -0.0148271, upper bound: 0.0148284
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 8.31
Output dim: 1, lower bound: -0.0148675, upper bound: 0.0148686
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 8.31
Output dim: 1, lower bound: -0.0148757, upper bound: 0.0148683
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 8.31
Output dim: 1, lower bound: -0.0150714, upper bound: 0.0150922
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 8.31
Output dim: 1, lower bound: -0.0150714, upper bound: 0.0150914
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 8.31
Output dim: 1, lower bound: -0.0150455, upper bound: 0.0150669
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 8.31
Output dim: 1, lower bound: -0.0150455, upper bound: 0.0150659
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 8.31
Output dim: 1, lower bound: -0.0147266, upper bound: 0.0147268
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 8.31
Output dim: 1, lower bound: -0.0147266, upper bound: 0.0147268
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 8.31
Output dim: 1, lower bound: -0.0150474, upper bound: 0.0150523
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 8.31
Output dim: 1, lower bound: -0.0150474, upper bound: 0.0150523

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0325615, 0.0325502
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397796, 0.0397826
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 90

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150050, upper bound: 0.0150127
time: 2.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150043, upper bound: 0.0150126
time: 2.68 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0325656, 0.0325463
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397786, 0.0397837
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 230

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 195

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150327, upper bound: 0.0150390
time: 2.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150327, upper bound: 0.0150390
time: 2.62 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0325180, 0.0325009
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397658, 0.0397703
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150542, upper bound: 0.0150621
time: 2.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150542, upper bound: 0.0150626
time: 2.62 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0325173, 0.0325031
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397663, 0.0397701
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 130

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 131

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0148585, upper bound: 0.0148637
time: 2.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0148585, upper bound: 0.0148637
time: 2.16 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0324663, 0.0324692
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397560, 0.0397553
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 203

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0146083, upper bound: 0.0146123
time: 2.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0146083, upper bound: 0.0146123
time: 2.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0324694, 0.0324659
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397551, 0.0397562
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 130

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0145816, upper bound: 0.0145881
time: 1.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0145816, upper bound: 0.0145880
time: 1.85 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0325316, 0.0325521
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397768, 0.0397714
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.05 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 4.85 + 595.61 = 600.46 seconds
