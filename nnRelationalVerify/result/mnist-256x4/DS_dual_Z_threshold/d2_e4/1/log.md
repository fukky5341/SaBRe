## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.020780391999999998


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893)
1: (-0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023)
2: (-0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883)
3: (-0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549)
4: (0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712)
5: (-0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050)
6: (0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164)
7: (-0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0361911, 0.0361911)
8: (-0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250)
9: (-0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.39 + 5.30 = 7.69 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.0221068, upper bound: 0.0221068

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0220997, upper bound: 0.0220997
time: 4.12 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0220997, upper bound: 0.0220997
time: 4.20 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 8.57 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 8.57
Output dim: 6, lower bound: -0.0220997, upper bound: 0.0220997
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 8.57
Output dim: 6, lower bound: -0.0220997, upper bound: 0.0220997

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0359698, 0.0359686
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 2.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0220676, upper bound: 0.0220667
time: 3.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0220667, upper bound: 0.0220676
time: 3.46 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0359686, 0.0359698
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 2.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0220676, upper bound: 0.0220667
time: 3.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0220667, upper bound: 0.0220676
time: 4.15 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 10.46 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 10.46
Output dim: 6, lower bound: -0.0220676, upper bound: 0.0220667
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 10.46
Output dim: 6, lower bound: -0.0220667, upper bound: 0.0220676
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 10.46
Output dim: 6, lower bound: -0.0220676, upper bound: 0.0220667
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 10.46
Output dim: 6, lower bound: -0.0220667, upper bound: 0.0220676

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0358536, 0.0358516
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 2.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0215573, upper bound: 0.0215573
time: 3.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0215573, upper bound: 0.0215573
time: 3.06 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0358471, 0.0358524
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 2.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0215569, upper bound: 0.0215575
time: 3.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0215569, upper bound: 0.0215575
time: 4.21 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0358524, 0.0358471
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 2.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0215575, upper bound: 0.0215569
time: 4.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0215575, upper bound: 0.0215569
time: 4.01 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0358516, 0.0358536
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 2.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0215573, upper bound: 0.0215573
time: 3.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0215573, upper bound: 0.0215573
time: 3.53 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 9.20 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 9.20
Output dim: 6, lower bound: -0.0215573, upper bound: 0.0215573
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 9.20
Output dim: 6, lower bound: -0.0215573, upper bound: 0.0215573
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 9.20
Output dim: 6, lower bound: -0.0215569, upper bound: 0.0215575
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 9.20
Output dim: 6, lower bound: -0.0215569, upper bound: 0.0215575
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 9.20
Output dim: 6, lower bound: -0.0215575, upper bound: 0.0215569
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 9.20
Output dim: 6, lower bound: -0.0215575, upper bound: 0.0215569
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 9.20
Output dim: 6, lower bound: -0.0215573, upper bound: 0.0215573
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 9.20
Output dim: 6, lower bound: -0.0215573, upper bound: 0.0215573

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0358191, 0.0358082
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 2.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0215389, upper bound: 0.0215390
time: 3.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0215388, upper bound: 0.0215391
time: 4.29 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0358101, 0.0358144
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 2.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0215389, upper bound: 0.0215391
time: 3.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0215388, upper bound: 0.0215390
time: 3.87 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0358108, 0.0358089
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 2.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0215386, upper bound: 0.0215392
time: 3.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0215386, upper bound: 0.0215392
time: 3.60 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0358037, 0.0358170
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 2.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0215386, upper bound: 0.0215391
time: 4.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0215386, upper bound: 0.0215392
time: 3.74 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0358170, 0.0358037
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 2.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0215392, upper bound: 0.0215386
time: 3.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0215392, upper bound: 0.0215386
time: 3.44 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0358089, 0.0358108
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 1.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0215392, upper bound: 0.0215386
time: 3.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0215392, upper bound: 0.0215386
time: 3.60 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0358144, 0.0358101
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 1.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0215391, upper bound: 0.0215389
time: 3.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0215391, upper bound: 0.0215389
time: 3.68 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0358082, 0.0358191
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 1.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0215391, upper bound: 0.0215389
time: 3.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0215391, upper bound: 0.0215389
time: 4.35 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 9.99 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 9.99
Output dim: 6, lower bound: -0.0215389, upper bound: 0.0215390
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 9.99
Output dim: 6, lower bound: -0.0215388, upper bound: 0.0215391
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 9.99
Output dim: 6, lower bound: -0.0215389, upper bound: 0.0215391
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 9.99
Output dim: 6, lower bound: -0.0215388, upper bound: 0.0215390
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 9.99
Output dim: 6, lower bound: -0.0215386, upper bound: 0.0215392
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 9.99
Output dim: 6, lower bound: -0.0215386, upper bound: 0.0215392
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 9.99
Output dim: 6, lower bound: -0.0215386, upper bound: 0.0215391
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 9.99
Output dim: 6, lower bound: -0.0215386, upper bound: 0.0215392
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 9.99
Output dim: 6, lower bound: -0.0215392, upper bound: 0.0215386
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 9.99
Output dim: 6, lower bound: -0.0215392, upper bound: 0.0215386
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 9.99
Output dim: 6, lower bound: -0.0215392, upper bound: 0.0215386
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 9.99
Output dim: 6, lower bound: -0.0215392, upper bound: 0.0215386
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 9.99
Output dim: 6, lower bound: -0.0215391, upper bound: 0.0215389
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 9.99
Output dim: 6, lower bound: -0.0215391, upper bound: 0.0215389
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 9.99
Output dim: 6, lower bound: -0.0215391, upper bound: 0.0215389
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 9.99
Output dim: 6, lower bound: -0.0215391, upper bound: 0.0215389

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0357965, 0.0357873
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 1.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0214380, upper bound: 0.0213767
time: 3.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213687, upper bound: 0.0214382
time: 3.67 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0358157, 0.0357855
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 1.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0214379, upper bound: 0.0213771
time: 3.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213687, upper bound: 0.0214383
time: 3.64 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0357875, 0.0357950
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 1.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0214380, upper bound: 0.0213767
time: 3.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213687, upper bound: 0.0214382
time: 3.38 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0358050, 0.0357918
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 1.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0214379, upper bound: 0.0213771
time: 3.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213687, upper bound: 0.0214382
time: 3.51 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0357881, 0.0357945
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 1.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0214376, upper bound: 0.0213761
time: 3.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213692, upper bound: 0.0214384
time: 3.32 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0358030, 0.0357863
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 1.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0214376, upper bound: 0.0213763
time: 3.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213692, upper bound: 0.0214384
time: 7.73 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0357810, 0.0358049
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 1.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0214376, upper bound: 0.0213761
time: 3.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213692, upper bound: 0.0214384
time: 3.62 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0357953, 0.0357943
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 1.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0214376, upper bound: 0.0213763
time: 3.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213692, upper bound: 0.0214384
time: 8.33 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0357943, 0.0357953
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 1.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0214384, upper bound: 0.0213692
time: 3.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213763, upper bound: 0.0214376
time: 2.68 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0358049, 0.0357810
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 1.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0214384, upper bound: 0.0213692
time: 3.97 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213761, upper bound: 0.0214376
time: 3.67 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0357863, 0.0358030
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 1.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0214384, upper bound: 0.0213692
time: 3.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213763, upper bound: 0.0214376
time: 2.67 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0357944, 0.0357881
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 1.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0214384, upper bound: 0.0213692
time: 3.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213761, upper bound: 0.0214376
time: 3.38 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0357918, 0.0358050
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 1.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0214382, upper bound: 0.0213687
time: 3.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213771, upper bound: 0.0214379
time: 3.67 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0357950, 0.0357875
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 1.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0214382, upper bound: 0.0213687
time: 3.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213767, upper bound: 0.0214380
time: 3.01 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0357855, 0.0358157
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 1.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0214382, upper bound: 0.0213687
time: 3.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213771, upper bound: 0.0214379
time: 3.73 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0357873, 0.0357965
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 1.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0214382, upper bound: 0.0213687
time: 3.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213767, upper bound: 0.0214380
time: 3.13 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 8.40 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 8.40
Output dim: 6, lower bound: -0.0214380, upper bound: 0.0213767
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 8.40
Output dim: 6, lower bound: -0.0213687, upper bound: 0.0214382
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 8.40
Output dim: 6, lower bound: -0.0214379, upper bound: 0.0213771
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 8.40
Output dim: 6, lower bound: -0.0213687, upper bound: 0.0214383
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 8.40
Output dim: 6, lower bound: -0.0214380, upper bound: 0.0213767
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 8.40
Output dim: 6, lower bound: -0.0213687, upper bound: 0.0214382
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 8.40
Output dim: 6, lower bound: -0.0214379, upper bound: 0.0213771
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 8.40
Output dim: 6, lower bound: -0.0213687, upper bound: 0.0214382
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 8.40
Output dim: 6, lower bound: -0.0214376, upper bound: 0.0213761
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 8.40
Output dim: 6, lower bound: -0.0213692, upper bound: 0.0214384
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 8.40
Output dim: 6, lower bound: -0.0214376, upper bound: 0.0213763
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 8.40
Output dim: 6, lower bound: -0.0213692, upper bound: 0.0214384
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 8.40
Output dim: 6, lower bound: -0.0214376, upper bound: 0.0213761
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 8.40
Output dim: 6, lower bound: -0.0213692, upper bound: 0.0214384
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 8.40
Output dim: 6, lower bound: -0.0214376, upper bound: 0.0213763
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 8.40
Output dim: 6, lower bound: -0.0213692, upper bound: 0.0214384
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 8.40
Output dim: 6, lower bound: -0.0214384, upper bound: 0.0213692
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 8.40
Output dim: 6, lower bound: -0.0213763, upper bound: 0.0214376
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 8.40
Output dim: 6, lower bound: -0.0214384, upper bound: 0.0213692
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 8.40
Output dim: 6, lower bound: -0.0213761, upper bound: 0.0214376
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 8.40
Output dim: 6, lower bound: -0.0214384, upper bound: 0.0213692
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 8.40
Output dim: 6, lower bound: -0.0213763, upper bound: 0.0214376
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 8.40
Output dim: 6, lower bound: -0.0214384, upper bound: 0.0213692
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 8.40
Output dim: 6, lower bound: -0.0213761, upper bound: 0.0214376
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 8.40
Output dim: 6, lower bound: -0.0214382, upper bound: 0.0213687
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 8.40
Output dim: 6, lower bound: -0.0213771, upper bound: 0.0214379
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 8.40
Output dim: 6, lower bound: -0.0214382, upper bound: 0.0213687
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 8.40
Output dim: 6, lower bound: -0.0213767, upper bound: 0.0214380
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 8.40
Output dim: 6, lower bound: -0.0214382, upper bound: 0.0213687
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 8.40
Output dim: 6, lower bound: -0.0213771, upper bound: 0.0214379
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 8.40
Output dim: 6, lower bound: -0.0214382, upper bound: 0.0213687
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 8.40
Output dim: 6, lower bound: -0.0213767, upper bound: 0.0214380

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0354681, 0.0352791
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 1.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213556, upper bound: 0.0212890
time: 3.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213478, upper bound: 0.0212937
time: 4.04 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0352883, 0.0354377
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 1.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0212858, upper bound: 0.0213482
time: 3.36 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0212818, upper bound: 0.0213559
time: 3.82 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0354864, 0.0352773
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 1.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213555, upper bound: 0.0212890
time: 3.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213477, upper bound: 0.0212943
time: 3.79 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0353075, 0.0354280
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 2.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0212858, upper bound: 0.0213482
time: 3.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0212819, upper bound: 0.0213560
time: 3.57 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0354602, 0.0352869
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 1.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213556, upper bound: 0.0212890
time: 3.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213478, upper bound: 0.0212938
time: 3.74 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0352793, 0.0354446
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 1.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0212858, upper bound: 0.0213482
time: 3.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0212818, upper bound: 0.0213559
time: 3.92 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0354775, 0.0352836
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 1.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213555, upper bound: 0.0212890
time: 3.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213477, upper bound: 0.0212943
time: 3.55 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0352968, 0.0354350
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 1.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0212858, upper bound: 0.0213482
time: 3.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0212819, upper bound: 0.0213560
time: 3.94 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0354456, 0.0352863
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 2.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213553, upper bound: 0.0212888
time: 3.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213475, upper bound: 0.0212933
time: 4.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0352799, 0.0354597
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 1.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0212863, upper bound: 0.0213482
time: 3.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0212822, upper bound: 0.0213561
time: 3.57 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0354600, 0.0352781
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 1.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213553, upper bound: 0.0212888
time: 3.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213476, upper bound: 0.0212936
time: 3.19 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0352948, 0.0354448
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 1.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0212863, upper bound: 0.0213482
time: 3.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0212823, upper bound: 0.0213561
time: 3.56 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0354381, 0.0352967
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 1.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213553, upper bound: 0.0212887
time: 3.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213475, upper bound: 0.0212933
time: 3.39 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0352729, 0.0354686
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 1.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0212863, upper bound: 0.0213482
time: 3.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0212822, upper bound: 0.0213561
time: 3.31 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0354524, 0.0352862
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 1.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213553, upper bound: 0.0212888
time: 4.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213476, upper bound: 0.0212936
time: 3.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0352872, 0.0354528
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 1.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0212863, upper bound: 0.0213482
time: 4.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0212823, upper bound: 0.0213561
time: 3.64 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0354528, 0.0352872
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 1.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213561, upper bound: 0.0212823
time: 3.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213482, upper bound: 0.0212863
time: 3.33 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0352862, 0.0354524
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 1.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0212936, upper bound: 0.0213476
time: 4.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0212888, upper bound: 0.0213553
time: 3.85 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0354686, 0.0352729
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 2.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213561, upper bound: 0.0212822
time: 3.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213482, upper bound: 0.0212863
time: 3.66 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0352967, 0.0354381
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 2.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0212933, upper bound: 0.0213475
time: 3.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0212819, upper bound: 0.0213553
time: 3.72 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0354448, 0.0352948
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 1.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213561, upper bound: 0.0212823
time: 3.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213482, upper bound: 0.0212863
time: 3.58 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0352781, 0.0354600
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 1.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0212936, upper bound: 0.0213476
time: 4.11 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0212888, upper bound: 0.0213553
time: 3.53 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0354597, 0.0352799
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 1.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213561, upper bound: 0.0212822
time: 3.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213482, upper bound: 0.0212863
time: 3.60 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0352863, 0.0354456
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 1.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0212933, upper bound: 0.0213475
time: 3.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0212887, upper bound: 0.0213553
time: 3.18 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0354350, 0.0352968
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 1.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213560, upper bound: 0.0212819
time: 3.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213482, upper bound: 0.0212858
time: 3.87 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0352836, 0.0354775
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 1.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0212943, upper bound: 0.0213477
time: 3.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0212890, upper bound: 0.0213555
time: 3.50 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0354446, 0.0352793
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 1.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213559, upper bound: 0.0212818
time: 3.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213482, upper bound: 0.0212858
time: 3.37 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0352869, 0.0354602
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 1.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0212937, upper bound: 0.0213477
time: 3.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0212889, upper bound: 0.0213556
time: 3.40 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0354280, 0.0353075
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 1.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213560, upper bound: 0.0212819
time: 3.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213482, upper bound: 0.0212858
time: 3.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0352773, 0.0354864
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 1.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0212943, upper bound: 0.0213477
time: 3.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0212890, upper bound: 0.0213556
time: 3.54 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0354377, 0.0352883
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 1.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213559, upper bound: 0.0212818
time: 8.90 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213482, upper bound: 0.0212858
time: 4.00 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0352791, 0.0354681
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 1.83 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 7.69 + 592.92 = 600.61 seconds
