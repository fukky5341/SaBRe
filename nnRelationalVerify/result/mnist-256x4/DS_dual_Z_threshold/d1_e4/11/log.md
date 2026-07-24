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
execution time: IAR + RelationalAnalysis = 1.72 + 3.54 = 5.26 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.0153569, upper bound: 0.0153569

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0152657, upper bound: 0.0152657
time: 2.36 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0152657, upper bound: 0.0152657
time: 2.19 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 4.74 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 4.74
Output dim: 1, lower bound: -0.0152657, upper bound: 0.0152657
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 4.74
Output dim: 1, lower bound: -0.0152657, upper bound: 0.0152657

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0327516, 0.0327524
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0398300, 0.0398298
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 251

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0152268, upper bound: 0.0152269
time: 2.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0152268, upper bound: 0.0152268
time: 2.06 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0327574, 0.0327516
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0398298, 0.0398314
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 251

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0152268, upper bound: 0.0152269
time: 2.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0152269, upper bound: 0.0152268
time: 2.30 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 6.20 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 6.20
Output dim: 1, lower bound: -0.0152268, upper bound: 0.0152269
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 6.20
Output dim: 1, lower bound: -0.0152268, upper bound: 0.0152268
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 6.20
Output dim: 1, lower bound: -0.0152268, upper bound: 0.0152269
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 6.20
Output dim: 1, lower bound: -0.0152269, upper bound: 0.0152268

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0326593, 0.0326597
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0398067, 0.0398066
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151542, upper bound: 0.0151680
time: 3.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151678, upper bound: 0.0151541
time: 2.45 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0326589, 0.0326592
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0398066, 0.0398065
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151541, upper bound: 0.0151678
time: 2.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151541, upper bound: 0.0151542
time: 2.31 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0326653, 0.0326589
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0398065, 0.0398083
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151542, upper bound: 0.0151680
time: 4.00 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151678, upper bound: 0.0151541
time: 2.40 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0326648, 0.0326593
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0398066, 0.0398081
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151541, upper bound: 0.0151677
time: 2.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151680, upper bound: 0.0151542
time: 2.33 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 6.63 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 6.63
Output dim: 1, lower bound: -0.0151542, upper bound: 0.0151680
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 6.63
Output dim: 1, lower bound: -0.0151678, upper bound: 0.0151541
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 6.63
Output dim: 1, lower bound: -0.0151541, upper bound: 0.0151678
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 6.63
Output dim: 1, lower bound: -0.0151541, upper bound: 0.0151542
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 6.63
Output dim: 1, lower bound: -0.0151542, upper bound: 0.0151680
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 6.63
Output dim: 1, lower bound: -0.0151678, upper bound: 0.0151541
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 6.63
Output dim: 1, lower bound: -0.0151541, upper bound: 0.0151677
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 6.63
Output dim: 1, lower bound: -0.0151680, upper bound: 0.0151542

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0326519, 0.0326534
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0398051, 0.0398047
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151261, upper bound: 0.0151408
time: 2.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151275, upper bound: 0.0151396
time: 2.45 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0326531, 0.0326519
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0398047, 0.0398050
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151258, upper bound: 0.0151273
time: 2.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151406, upper bound: 0.0151257
time: 2.28 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0326512, 0.0326529
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0398049, 0.0398045
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151258, upper bound: 0.0151406
time: 2.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151258, upper bound: 0.0151394
time: 2.51 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0326526, 0.0326517
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0398046, 0.0398048
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151258, upper bound: 0.0151275
time: 2.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151258, upper bound: 0.0151261
time: 2.47 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0326578, 0.0326526
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0398048, 0.0398063
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151258, upper bound: 0.0151408
time: 2.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151275, upper bound: 0.0151396
time: 2.33 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0326590, 0.0326512
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0398045, 0.0398066
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151394, upper bound: 0.0151273
time: 2.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151406, upper bound: 0.0151257
time: 2.30 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0326571, 0.0326531
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0398050, 0.0398061
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151258, upper bound: 0.0151406
time: 2.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151258, upper bound: 0.0151394
time: 2.56 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0326585, 0.0326519
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0398047, 0.0398065
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151396, upper bound: 0.0151275
time: 2.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151258, upper bound: 0.0151261
time: 2.75 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 7.45 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 7.45
Output dim: 1, lower bound: -0.0151261, upper bound: 0.0151408
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 7.45
Output dim: 1, lower bound: -0.0151275, upper bound: 0.0151396
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 7.45
Output dim: 1, lower bound: -0.0151258, upper bound: 0.0151273
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 7.45
Output dim: 1, lower bound: -0.0151406, upper bound: 0.0151257
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 7.45
Output dim: 1, lower bound: -0.0151258, upper bound: 0.0151406
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 7.45
Output dim: 1, lower bound: -0.0151258, upper bound: 0.0151394
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 7.45
Output dim: 1, lower bound: -0.0151258, upper bound: 0.0151275
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 7.45
Output dim: 1, lower bound: -0.0151258, upper bound: 0.0151261
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 7.45
Output dim: 1, lower bound: -0.0151258, upper bound: 0.0151408
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 7.45
Output dim: 1, lower bound: -0.0151275, upper bound: 0.0151396
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 7.45
Output dim: 1, lower bound: -0.0151394, upper bound: 0.0151273
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 7.45
Output dim: 1, lower bound: -0.0151406, upper bound: 0.0151257
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 7.45
Output dim: 1, lower bound: -0.0151258, upper bound: 0.0151406
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 7.45
Output dim: 1, lower bound: -0.0151258, upper bound: 0.0151394
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 7.45
Output dim: 1, lower bound: -0.0151396, upper bound: 0.0151275
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 7.45
Output dim: 1, lower bound: -0.0151258, upper bound: 0.0151261

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0326066, 0.0326107
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397954, 0.0397943
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 131

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149572, upper bound: 0.0149690
time: 2.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149572, upper bound: 0.0149690
time: 2.46 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0326093, 0.0326067
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397943, 0.0397950
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 131

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149577, upper bound: 0.0149685
time: 2.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149577, upper bound: 0.0149685
time: 2.38 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0326077, 0.0326092
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397950, 0.0397946
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 131

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149687, upper bound: 0.0149572
time: 2.09 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149687, upper bound: 0.0149572
time: 2.04 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0326104, 0.0326053
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397940, 0.0397953
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 131

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149692, upper bound: 0.0149567
time: 2.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149692, upper bound: 0.0149567
time: 2.37 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0326048, 0.0326102
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397953, 0.0397938
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 131

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149567, upper bound: 0.0149692
time: 2.13 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149567, upper bound: 0.0149692
time: 2.14 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0326085, 0.0326073
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397945, 0.0397948
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 131

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149572, upper bound: 0.0149687
time: 2.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149572, upper bound: 0.0149687
time: 2.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0326062, 0.0326090
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397950, 0.0397942
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 131

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149685, upper bound: 0.0149577
time: 2.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149685, upper bound: 0.0149577
time: 2.15 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0326099, 0.0326061
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397942, 0.0397952
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 131

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149690, upper bound: 0.0149572
time: 2.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149690, upper bound: 0.0149572
time: 2.33 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0326127, 0.0326099
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397952, 0.0397960
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 131

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149572, upper bound: 0.0149690
time: 2.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149567, upper bound: 0.0149690
time: 2.34 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0326154, 0.0326062
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397942, 0.0397967
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 131

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149577, upper bound: 0.0149685
time: 2.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149577, upper bound: 0.0149685
time: 2.24 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0326139, 0.0326085
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397948, 0.0397963
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 131

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149687, upper bound: 0.0149572
time: 2.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149687, upper bound: 0.0149572
time: 2.12 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0326165, 0.0326048
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397938, 0.0397970
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 131

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149692, upper bound: 0.0149567
time: 2.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149692, upper bound: 0.0149567
time: 2.45 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0326109, 0.0326104
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397953, 0.0397955
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 131

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149567, upper bound: 0.0149692
time: 2.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149567, upper bound: 0.0149692
time: 2.08 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0326146, 0.0326077
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397946, 0.0397965
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 131

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149567, upper bound: 0.0149687
time: 2.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149567, upper bound: 0.0149687
time: 2.47 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0326123, 0.0326093
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397950, 0.0397959
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 131

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149685, upper bound: 0.0149577
time: 2.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149685, upper bound: 0.0149577
time: 2.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0326160, 0.0326066
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397943, 0.0397969
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 131

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149690, upper bound: 0.0149572
time: 2.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149690, upper bound: 0.0149572
time: 2.27 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 8.11 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 8.11
Output dim: 1, lower bound: -0.0149572, upper bound: 0.0149690
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 8.11
Output dim: 1, lower bound: -0.0149572, upper bound: 0.0149690
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 8.11
Output dim: 1, lower bound: -0.0149577, upper bound: 0.0149685
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 8.11
Output dim: 1, lower bound: -0.0149577, upper bound: 0.0149685
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 8.11
Output dim: 1, lower bound: -0.0149687, upper bound: 0.0149572
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 8.11
Output dim: 1, lower bound: -0.0149687, upper bound: 0.0149572
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 8.11
Output dim: 1, lower bound: -0.0149692, upper bound: 0.0149567
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 8.11
Output dim: 1, lower bound: -0.0149692, upper bound: 0.0149567
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 8.11
Output dim: 1, lower bound: -0.0149567, upper bound: 0.0149692
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 8.11
Output dim: 1, lower bound: -0.0149567, upper bound: 0.0149692
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 8.11
Output dim: 1, lower bound: -0.0149572, upper bound: 0.0149687
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 8.11
Output dim: 1, lower bound: -0.0149572, upper bound: 0.0149687
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 8.11
Output dim: 1, lower bound: -0.0149685, upper bound: 0.0149577
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 8.11
Output dim: 1, lower bound: -0.0149685, upper bound: 0.0149577
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 8.11
Output dim: 1, lower bound: -0.0149690, upper bound: 0.0149572
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 8.11
Output dim: 1, lower bound: -0.0149690, upper bound: 0.0149572
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 8.11
Output dim: 1, lower bound: -0.0149572, upper bound: 0.0149690
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 8.11
Output dim: 1, lower bound: -0.0149567, upper bound: 0.0149690
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 8.11
Output dim: 1, lower bound: -0.0149577, upper bound: 0.0149685
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 8.11
Output dim: 1, lower bound: -0.0149577, upper bound: 0.0149685
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 8.11
Output dim: 1, lower bound: -0.0149687, upper bound: 0.0149572
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 8.11
Output dim: 1, lower bound: -0.0149687, upper bound: 0.0149572
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 8.11
Output dim: 1, lower bound: -0.0149692, upper bound: 0.0149567
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 8.11
Output dim: 1, lower bound: -0.0149692, upper bound: 0.0149567
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 8.11
Output dim: 1, lower bound: -0.0149567, upper bound: 0.0149692
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 8.11
Output dim: 1, lower bound: -0.0149567, upper bound: 0.0149692
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 8.11
Output dim: 1, lower bound: -0.0149567, upper bound: 0.0149687
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 8.11
Output dim: 1, lower bound: -0.0149567, upper bound: 0.0149687
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 8.11
Output dim: 1, lower bound: -0.0149685, upper bound: 0.0149577
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 8.11
Output dim: 1, lower bound: -0.0149685, upper bound: 0.0149577
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 8.11
Output dim: 1, lower bound: -0.0149690, upper bound: 0.0149572
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 8.11
Output dim: 1, lower bound: -0.0149690, upper bound: 0.0149572

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0326037, 0.0326088
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397949, 0.0397936
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Candidate
type: DSZ, layer: 1, pos: 134

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149451, upper bound: 0.0149575
time: 2.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149457, upper bound: 0.0149567
time: 2.26 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0326047, 0.0326107
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397954, 0.0397938
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Candidate
type: DSZ, layer: 1, pos: 134

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149444, upper bound: 0.0149575
time: 2.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149457, upper bound: 0.0149567
time: 2.16 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0326064, 0.0326048
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397938, 0.0397943
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Candidate
type: DSZ, layer: 1, pos: 134

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149457, upper bound: 0.0149570
time: 2.09 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149461, upper bound: 0.0149561
time: 2.02 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0326073, 0.0326067
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397943, 0.0397945
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Candidate
type: DSZ, layer: 1, pos: 134

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149457, upper bound: 0.0149570
time: 2.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149461, upper bound: 0.0149561
time: 1.97 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0326048, 0.0326073
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397945, 0.0397939
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Candidate
type: DSZ, layer: 1, pos: 134

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149564, upper bound: 0.0149457
time: 2.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149571, upper bound: 0.0149450
time: 2.21 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0326058, 0.0326092
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397950, 0.0397941
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Candidate
type: DSZ, layer: 1, pos: 134

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149564, upper bound: 0.0149457
time: 2.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149571, upper bound: 0.0149450
time: 2.15 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0326074, 0.0326034
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397935, 0.0397945
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Candidate
type: DSZ, layer: 1, pos: 134

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149569, upper bound: 0.0149452
time: 2.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149576, upper bound: 0.0149444
time: 2.26 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0326085, 0.0326053
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397940, 0.0397948
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Candidate
type: DSZ, layer: 1, pos: 134

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149569, upper bound: 0.0149452
time: 2.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149576, upper bound: 0.0149444
time: 2.26 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0326017, 0.0326083
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397948, 0.0397930
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Candidate
type: DSZ, layer: 1, pos: 134

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149444, upper bound: 0.0149576
time: 2.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149452, upper bound: 0.0149569
time: 2.26 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0326029, 0.0326102
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397953, 0.0397933
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Candidate
type: DSZ, layer: 1, pos: 134

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149444, upper bound: 0.0149576
time: 2.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149452, upper bound: 0.0149569
time: 2.23 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0326055, 0.0326054
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397940, 0.0397940
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Candidate
type: DSZ, layer: 1, pos: 134

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149444, upper bound: 0.0149571
time: 2.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149457, upper bound: 0.0149564
time: 2.39 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0326066, 0.0326073
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397945, 0.0397943
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Candidate
type: DSZ, layer: 1, pos: 134

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149450, upper bound: 0.0149571
time: 2.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149457, upper bound: 0.0149564
time: 2.35 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0326031, 0.0326071
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397945, 0.0397934
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Candidate
type: DSZ, layer: 1, pos: 134

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149561, upper bound: 0.0149461
time: 2.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149570, upper bound: 0.0149457
time: 2.04 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0326043, 0.0326090
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397950, 0.0397937
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Candidate
type: DSZ, layer: 1, pos: 134

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149561, upper bound: 0.0149461
time: 2.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149570, upper bound: 0.0149457
time: 1.99 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0326068, 0.0326042
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397937, 0.0397944
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Candidate
type: DSZ, layer: 1, pos: 134

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149567, upper bound: 0.0149457
time: 2.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149575, upper bound: 0.0149451
time: 2.40 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0326080, 0.0326061
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397942, 0.0397947
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Candidate
type: DSZ, layer: 1, pos: 134

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149567, upper bound: 0.0149457
time: 2.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149575, upper bound: 0.0149451
time: 2.29 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0326099, 0.0326080
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397947, 0.0397952
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Candidate
type: DSZ, layer: 1, pos: 134

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149451, upper bound: 0.0149575
time: 2.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149457, upper bound: 0.0149567
time: 2.27 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0326108, 0.0326099
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397952, 0.0397955
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Candidate
type: DSZ, layer: 1, pos: 134

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149451, upper bound: 0.0149575
time: 2.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149457, upper bound: 0.0149567
time: 2.35 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0326125, 0.0326043
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397937, 0.0397959
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Candidate
type: DSZ, layer: 1, pos: 134

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149457, upper bound: 0.0149570
time: 2.12 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149461, upper bound: 0.0149561
time: 2.14 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0326134, 0.0326062
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397942, 0.0397962
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Candidate
type: DSZ, layer: 1, pos: 134

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149457, upper bound: 0.0149570
time: 2.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149461, upper bound: 0.0149561
time: 1.99 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0326109, 0.0326066
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397943, 0.0397955
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Candidate
type: DSZ, layer: 1, pos: 134

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149564, upper bound: 0.0149457
time: 2.12 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149571, upper bound: 0.0149450
time: 2.16 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0326119, 0.0326085
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397948, 0.0397958
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Candidate
type: DSZ, layer: 1, pos: 134

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149564, upper bound: 0.0149457
time: 2.08 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149571, upper bound: 0.0149450
time: 2.13 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0326135, 0.0326029
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397933, 0.0397962
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Candidate
type: DSZ, layer: 1, pos: 134

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149569, upper bound: 0.0149452
time: 2.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149576, upper bound: 0.0149444
time: 2.42 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0326146, 0.0326048
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397938, 0.0397965
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Candidate
type: DSZ, layer: 1, pos: 134

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149569, upper bound: 0.0149452
time: 2.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149576, upper bound: 0.0149444
time: 2.31 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0326078, 0.0326085
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397948, 0.0397947
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Candidate
type: DSZ, layer: 1, pos: 134

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149444, upper bound: 0.0149576
time: 2.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149452, upper bound: 0.0149569
time: 2.42 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0326090, 0.0326104
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397953, 0.0397950
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Candidate
type: DSZ, layer: 1, pos: 134

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149444, upper bound: 0.0149576
time: 2.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149452, upper bound: 0.0149569
time: 2.40 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0326116, 0.0326058
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397941, 0.0397957
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Candidate
type: DSZ, layer: 1, pos: 134

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149444, upper bound: 0.0149571
time: 2.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149457, upper bound: 0.0149563
time: 2.50 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0326127, 0.0326077
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397946, 0.0397960
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Candidate
type: DSZ, layer: 1, pos: 134

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149444, upper bound: 0.0149571
time: 2.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149457, upper bound: 0.0149564
time: 2.45 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0326092, 0.0326073
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397945, 0.0397951
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Candidate
type: DSZ, layer: 1, pos: 134

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149561, upper bound: 0.0149461
time: 2.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149570, upper bound: 0.0149457
time: 2.52 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0326104, 0.0326093
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397950, 0.0397954
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Candidate
type: DSZ, layer: 1, pos: 134

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149561, upper bound: 0.0149461
time: 2.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149570, upper bound: 0.0149457
time: 2.49 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0326129, 0.0326047
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397938, 0.0397961
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Candidate
type: DSZ, layer: 1, pos: 134

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149567, upper bound: 0.0149457
time: 2.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149575, upper bound: 0.0149451
time: 2.28 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0326141, 0.0326066
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397943, 0.0397964
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Candidate
type: DSZ, layer: 1, pos: 134

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149567, upper bound: 0.0149457
time: 2.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149575, upper bound: 0.0149451
time: 2.35 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 6.64 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.64
Output dim: 1, lower bound: -0.0149451, upper bound: 0.0149575
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.64
Output dim: 1, lower bound: -0.0149457, upper bound: 0.0149567
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.64
Output dim: 1, lower bound: -0.0149444, upper bound: 0.0149575
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.64
Output dim: 1, lower bound: -0.0149457, upper bound: 0.0149567
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.64
Output dim: 1, lower bound: -0.0149457, upper bound: 0.0149570
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.64
Output dim: 1, lower bound: -0.0149461, upper bound: 0.0149561
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.64
Output dim: 1, lower bound: -0.0149457, upper bound: 0.0149570
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.64
Output dim: 1, lower bound: -0.0149461, upper bound: 0.0149561
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.64
Output dim: 1, lower bound: -0.0149564, upper bound: 0.0149457
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.64
Output dim: 1, lower bound: -0.0149571, upper bound: 0.0149450
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.64
Output dim: 1, lower bound: -0.0149564, upper bound: 0.0149457
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.64
Output dim: 1, lower bound: -0.0149571, upper bound: 0.0149450
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.64
Output dim: 1, lower bound: -0.0149569, upper bound: 0.0149452
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.64
Output dim: 1, lower bound: -0.0149576, upper bound: 0.0149444
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.64
Output dim: 1, lower bound: -0.0149569, upper bound: 0.0149452
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.64
Output dim: 1, lower bound: -0.0149576, upper bound: 0.0149444
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.64
Output dim: 1, lower bound: -0.0149444, upper bound: 0.0149576
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.64
Output dim: 1, lower bound: -0.0149452, upper bound: 0.0149569
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.64
Output dim: 1, lower bound: -0.0149444, upper bound: 0.0149576
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.64
Output dim: 1, lower bound: -0.0149452, upper bound: 0.0149569
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.64
Output dim: 1, lower bound: -0.0149444, upper bound: 0.0149571
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.64
Output dim: 1, lower bound: -0.0149457, upper bound: 0.0149564
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.64
Output dim: 1, lower bound: -0.0149450, upper bound: 0.0149571
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.64
Output dim: 1, lower bound: -0.0149457, upper bound: 0.0149564
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.64
Output dim: 1, lower bound: -0.0149561, upper bound: 0.0149461
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.64
Output dim: 1, lower bound: -0.0149570, upper bound: 0.0149457
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.64
Output dim: 1, lower bound: -0.0149561, upper bound: 0.0149461
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.64
Output dim: 1, lower bound: -0.0149570, upper bound: 0.0149457
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.64
Output dim: 1, lower bound: -0.0149567, upper bound: 0.0149457
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.64
Output dim: 1, lower bound: -0.0149575, upper bound: 0.0149451
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.64
Output dim: 1, lower bound: -0.0149567, upper bound: 0.0149457
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.64
Output dim: 1, lower bound: -0.0149575, upper bound: 0.0149451
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.64
Output dim: 1, lower bound: -0.0149451, upper bound: 0.0149575
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.64
Output dim: 1, lower bound: -0.0149457, upper bound: 0.0149567
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.64
Output dim: 1, lower bound: -0.0149451, upper bound: 0.0149575
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.64
Output dim: 1, lower bound: -0.0149457, upper bound: 0.0149567
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.64
Output dim: 1, lower bound: -0.0149457, upper bound: 0.0149570
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.64
Output dim: 1, lower bound: -0.0149461, upper bound: 0.0149561
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.64
Output dim: 1, lower bound: -0.0149457, upper bound: 0.0149570
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.64
Output dim: 1, lower bound: -0.0149461, upper bound: 0.0149561
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.64
Output dim: 1, lower bound: -0.0149564, upper bound: 0.0149457
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.64
Output dim: 1, lower bound: -0.0149571, upper bound: 0.0149450
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.64
Output dim: 1, lower bound: -0.0149564, upper bound: 0.0149457
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.64
Output dim: 1, lower bound: -0.0149571, upper bound: 0.0149450
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.64
Output dim: 1, lower bound: -0.0149569, upper bound: 0.0149452
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.64
Output dim: 1, lower bound: -0.0149576, upper bound: 0.0149444
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.64
Output dim: 1, lower bound: -0.0149569, upper bound: 0.0149452
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.64
Output dim: 1, lower bound: -0.0149576, upper bound: 0.0149444
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.64
Output dim: 1, lower bound: -0.0149444, upper bound: 0.0149576
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.64
Output dim: 1, lower bound: -0.0149452, upper bound: 0.0149569
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.64
Output dim: 1, lower bound: -0.0149444, upper bound: 0.0149576
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.64
Output dim: 1, lower bound: -0.0149452, upper bound: 0.0149569
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.64
Output dim: 1, lower bound: -0.0149444, upper bound: 0.0149571
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.64
Output dim: 1, lower bound: -0.0149457, upper bound: 0.0149563
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.64
Output dim: 1, lower bound: -0.0149444, upper bound: 0.0149571
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.64
Output dim: 1, lower bound: -0.0149457, upper bound: 0.0149564
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.64
Output dim: 1, lower bound: -0.0149561, upper bound: 0.0149461
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.64
Output dim: 1, lower bound: -0.0149570, upper bound: 0.0149457
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.64
Output dim: 1, lower bound: -0.0149561, upper bound: 0.0149461
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.64
Output dim: 1, lower bound: -0.0149570, upper bound: 0.0149457
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.64
Output dim: 1, lower bound: -0.0149567, upper bound: 0.0149457
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.64
Output dim: 1, lower bound: -0.0149575, upper bound: 0.0149451
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.64
Output dim: 1, lower bound: -0.0149567, upper bound: 0.0149457
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.64
Output dim: 1, lower bound: -0.0149575, upper bound: 0.0149451

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0325883, 0.0325923
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397920, 0.0397909
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Candidate
type: DSZ, layer: 1, pos: 229

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149171, upper bound: 0.0149384
time: 2.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149250, upper bound: 0.0149276
time: 2.33 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0325873, 0.0325913
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397917, 0.0397906
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Candidate
type: DSZ, layer: 1, pos: 229

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149182, upper bound: 0.0149375
time: 2.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149171, upper bound: 0.0149272
time: 2.44 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0325893, 0.0325942
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397925, 0.0397912
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Candidate
type: DSZ, layer: 1, pos: 229

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149171, upper bound: 0.0149384
time: 2.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149250, upper bound: 0.0149276
time: 2.34 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0325882, 0.0325932
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397922, 0.0397909
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Candidate
type: DSZ, layer: 1, pos: 229

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149182, upper bound: 0.0149375
time: 2.36 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149171, upper bound: 0.0149271
time: 2.44 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0325912, 0.0325883
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397909, 0.0397917
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Candidate
type: DSZ, layer: 1, pos: 229

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149171, upper bound: 0.0149378
time: 2.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149257, upper bound: 0.0149271
time: 2.42 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0325899, 0.0325874
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397907, 0.0397913
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Candidate
type: DSZ, layer: 1, pos: 229

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149171, upper bound: 0.0149368
time: 2.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149262, upper bound: 0.0149265
time: 2.48 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0325922, 0.0325902
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397914, 0.0397920
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Candidate
type: DSZ, layer: 1, pos: 229

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149171, upper bound: 0.0149378
time: 2.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149257, upper bound: 0.0149271
time: 2.43 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0325909, 0.0325893
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397912, 0.0397916
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Candidate
type: DSZ, layer: 1, pos: 229

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149185, upper bound: 0.0149368
time: 2.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149262, upper bound: 0.0149265
time: 2.57 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0325894, 0.0325908
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397916, 0.0397912
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Candidate
type: DSZ, layer: 1, pos: 229

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149270, upper bound: 0.0149257
time: 2.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149370, upper bound: 0.0149181
time: 2.30 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0325883, 0.0325899
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397913, 0.0397909
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Candidate
type: DSZ, layer: 1, pos: 229

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149274, upper bound: 0.0149250
time: 2.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149379, upper bound: 0.0149175
time: 2.47 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0325904, 0.0325927
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397921, 0.0397915
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Candidate
type: DSZ, layer: 1, pos: 229

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149270, upper bound: 0.0149257
time: 2.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149370, upper bound: 0.0149181
time: 2.31 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0325893, 0.0325918
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397918, 0.0397912
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Candidate
type: DSZ, layer: 1, pos: 229

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149274, upper bound: 0.0149250
time: 2.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149379, upper bound: 0.0149175
time: 2.39 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0325925, 0.0325869
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397905, 0.0397920
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Candidate
type: DSZ, layer: 1, pos: 229

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149171, upper bound: 0.0149251
time: 2.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149376, upper bound: 0.0149179
time: 2.38 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0325909, 0.0325859
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397903, 0.0397916
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Candidate
type: DSZ, layer: 1, pos: 229

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149171, upper bound: 0.0149242
time: 2.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149385, upper bound: 0.0149171
time: 2.48 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0325935, 0.0325888
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397911, 0.0397923
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Candidate
type: DSZ, layer: 1, pos: 229

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149171, upper bound: 0.0149251
time: 2.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149376, upper bound: 0.0149179
time: 2.35 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0325920, 0.0325878
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397908, 0.0397919
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Candidate
type: DSZ, layer: 1, pos: 229

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149171, upper bound: 0.0149241
time: 2.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149385, upper bound: 0.0149171
time: 2.40 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0325842, 0.0325918
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397919, 0.0397898
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Candidate
type: DSZ, layer: 1, pos: 229

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149171, upper bound: 0.0149384
time: 2.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149171, upper bound: 0.0149280
time: 2.57 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0325852, 0.0325935
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397923, 0.0397901
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Candidate
type: DSZ, layer: 1, pos: 229

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149171, upper bound: 0.0149376
time: 2.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149171, upper bound: 0.0149276
time: 2.23 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0325852, 0.0325937
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397924, 0.0397901
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Candidate
type: DSZ, layer: 1, pos: 229

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149171, upper bound: 0.0149384
time: 2.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149171, upper bound: 0.0149280
time: 2.64 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0325864, 0.0325954
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397928, 0.0397904
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Candidate
type: DSZ, layer: 1, pos: 229

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149171, upper bound: 0.0149376
time: 2.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149171, upper bound: 0.0149276
time: 2.34 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0325880, 0.0325889
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397911, 0.0397909
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Candidate
type: DSZ, layer: 1, pos: 229

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149171, upper bound: 0.0149379
time: 2.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149250, upper bound: 0.0149274
time: 2.31 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0325890, 0.0325899
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397913, 0.0397911
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Candidate
type: DSZ, layer: 1, pos: 229

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149171, upper bound: 0.0149370
time: 2.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149257, upper bound: 0.0149270
time: 2.36 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0325890, 0.0325908
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397916, 0.0397911
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Candidate
type: DSZ, layer: 1, pos: 229

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149171, upper bound: 0.0149379
time: 2.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149250, upper bound: 0.0149274
time: 2.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0325901, 0.0325918
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397918, 0.0397914
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Candidate
type: DSZ, layer: 1, pos: 229

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149171, upper bound: 0.0149370
time: 2.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149257, upper bound: 0.0149270
time: 2.37 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959
1: 0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687
2: -0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0325856, 0.0325906
3: -0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814
4: -0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126
5: -0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610
6: -0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640
7: -0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709
8: -0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0397916, 0.0397902
9: -0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511

Time for backsubstitution: 1.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Candidate
type: DSZ, layer: 1, pos: 229

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149265, upper bound: 0.0149262
time: 2.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149171, upper bound: 0.0149184
time: 2.68 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 6.69 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.69
Output dim: 1, lower bound: -0.0149171, upper bound: 0.0149384
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.69
Output dim: 1, lower bound: -0.0149250, upper bound: 0.0149276
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.69
Output dim: 1, lower bound: -0.0149182, upper bound: 0.0149375
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.69
Output dim: 1, lower bound: -0.0149171, upper bound: 0.0149272
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.69
Output dim: 1, lower bound: -0.0149171, upper bound: 0.0149384
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.69
Output dim: 1, lower bound: -0.0149250, upper bound: 0.0149276
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.69
Output dim: 1, lower bound: -0.0149182, upper bound: 0.0149375
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.69
Output dim: 1, lower bound: -0.0149171, upper bound: 0.0149271
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.69
Output dim: 1, lower bound: -0.0149171, upper bound: 0.0149378
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.69
Output dim: 1, lower bound: -0.0149257, upper bound: 0.0149271
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.69
Output dim: 1, lower bound: -0.0149171, upper bound: 0.0149368
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.69
Output dim: 1, lower bound: -0.0149262, upper bound: 0.0149265
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.69
Output dim: 1, lower bound: -0.0149171, upper bound: 0.0149378
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.69
Output dim: 1, lower bound: -0.0149257, upper bound: 0.0149271
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.69
Output dim: 1, lower bound: -0.0149185, upper bound: 0.0149368
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.69
Output dim: 1, lower bound: -0.0149262, upper bound: 0.0149265
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.69
Output dim: 1, lower bound: -0.0149270, upper bound: 0.0149257
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.69
Output dim: 1, lower bound: -0.0149370, upper bound: 0.0149181
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.69
Output dim: 1, lower bound: -0.0149274, upper bound: 0.0149250
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.69
Output dim: 1, lower bound: -0.0149379, upper bound: 0.0149175
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.69
Output dim: 1, lower bound: -0.0149270, upper bound: 0.0149257
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.69
Output dim: 1, lower bound: -0.0149370, upper bound: 0.0149181
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.69
Output dim: 1, lower bound: -0.0149274, upper bound: 0.0149250
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.69
Output dim: 1, lower bound: -0.0149379, upper bound: 0.0149175
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.69
Output dim: 1, lower bound: -0.0149171, upper bound: 0.0149251
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.69
Output dim: 1, lower bound: -0.0149376, upper bound: 0.0149179
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.69
Output dim: 1, lower bound: -0.0149171, upper bound: 0.0149242
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.69
Output dim: 1, lower bound: -0.0149385, upper bound: 0.0149171
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.69
Output dim: 1, lower bound: -0.0149171, upper bound: 0.0149251
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.69
Output dim: 1, lower bound: -0.0149376, upper bound: 0.0149179
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.69
Output dim: 1, lower bound: -0.0149171, upper bound: 0.0149241
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.69
Output dim: 1, lower bound: -0.0149385, upper bound: 0.0149171
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.69
Output dim: 1, lower bound: -0.0149171, upper bound: 0.0149384
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.69
Output dim: 1, lower bound: -0.0149171, upper bound: 0.0149280
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.69
Output dim: 1, lower bound: -0.0149171, upper bound: 0.0149376
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.69
Output dim: 1, lower bound: -0.0149171, upper bound: 0.0149276
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.69
Output dim: 1, lower bound: -0.0149171, upper bound: 0.0149384
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.69
Output dim: 1, lower bound: -0.0149171, upper bound: 0.0149280
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.69
Output dim: 1, lower bound: -0.0149171, upper bound: 0.0149376
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.69
Output dim: 1, lower bound: -0.0149171, upper bound: 0.0149276
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.69
Output dim: 1, lower bound: -0.0149171, upper bound: 0.0149379
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.69
Output dim: 1, lower bound: -0.0149250, upper bound: 0.0149274
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.69
Output dim: 1, lower bound: -0.0149171, upper bound: 0.0149370
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.69
Output dim: 1, lower bound: -0.0149257, upper bound: 0.0149270
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.69
Output dim: 1, lower bound: -0.0149171, upper bound: 0.0149379
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.69
Output dim: 1, lower bound: -0.0149250, upper bound: 0.0149274
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.69
Output dim: 1, lower bound: -0.0149171, upper bound: 0.0149370
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.69
Output dim: 1, lower bound: -0.0149257, upper bound: 0.0149270
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.69
Output dim: 1, lower bound: -0.0149265, upper bound: 0.0149262
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.69
Output dim: 1, lower bound: -0.0149171, upper bound: 0.0149184
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 1, lower bound: -0.0149570, upper bound: 0.0149457
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 1, lower bound: -0.0149561, upper bound: 0.0149461
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 1, lower bound: -0.0149570, upper bound: 0.0149457
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 1, lower bound: -0.0149567, upper bound: 0.0149457
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 1, lower bound: -0.0149575, upper bound: 0.0149451
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 1, lower bound: -0.0149567, upper bound: 0.0149457
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 1, lower bound: -0.0149575, upper bound: 0.0149451
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 1, lower bound: -0.0149451, upper bound: 0.0149575
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 1, lower bound: -0.0149457, upper bound: 0.0149567
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 1, lower bound: -0.0149451, upper bound: 0.0149575
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 1, lower bound: -0.0149457, upper bound: 0.0149567
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 1, lower bound: -0.0149457, upper bound: 0.0149570
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 1, lower bound: -0.0149461, upper bound: 0.0149561
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 1, lower bound: -0.0149457, upper bound: 0.0149570
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 1, lower bound: -0.0149461, upper bound: 0.0149561
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 1, lower bound: -0.0149564, upper bound: 0.0149457
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 1, lower bound: -0.0149571, upper bound: 0.0149450
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 1, lower bound: -0.0149564, upper bound: 0.0149457
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 1, lower bound: -0.0149571, upper bound: 0.0149450
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 1, lower bound: -0.0149569, upper bound: 0.0149452
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 1, lower bound: -0.0149576, upper bound: 0.0149444
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 1, lower bound: -0.0149569, upper bound: 0.0149452
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 1, lower bound: -0.0149576, upper bound: 0.0149444
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 1, lower bound: -0.0149444, upper bound: 0.0149576
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 1, lower bound: -0.0149452, upper bound: 0.0149569
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 1, lower bound: -0.0149444, upper bound: 0.0149576
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 1, lower bound: -0.0149452, upper bound: 0.0149569
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 1, lower bound: -0.0149444, upper bound: 0.0149571
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 1, lower bound: -0.0149457, upper bound: 0.0149563
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 1, lower bound: -0.0149444, upper bound: 0.0149571
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 1, lower bound: -0.0149457, upper bound: 0.0149564
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 1, lower bound: -0.0149561, upper bound: 0.0149461
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 1, lower bound: -0.0149570, upper bound: 0.0149457
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 1, lower bound: -0.0149561, upper bound: 0.0149461
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 1, lower bound: -0.0149570, upper bound: 0.0149457
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 1, lower bound: -0.0149567, upper bound: 0.0149457
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 1, lower bound: -0.0149575, upper bound: 0.0149451
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 1, lower bound: -0.0149567, upper bound: 0.0149457
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.69
Output dim: 1, lower bound: -0.0149575, upper bound: 0.0149451

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 5.26 + 597.81 = 603.07 seconds
