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
execution time: IAR + RelationalAnalysis = 0.81 + 5.04 = 5.85 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.0221068, upper bound: 0.0221068

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 188

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0220518, upper bound: 0.0220442
time: 2.68 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0220443, upper bound: 0.0220518
time: 3.56 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 6.25 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 6.25
Output dim: 6, lower bound: -0.0220518, upper bound: 0.0220442
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 6.25
Output dim: 6, lower bound: -0.0220443, upper bound: 0.0220518

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0361755, 0.0361750
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0216765, upper bound: 0.0216694
time: 3.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0216765, upper bound: 0.0216694
time: 3.48 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0361750, 0.0361755
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 199

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0219345, upper bound: 0.0218989
time: 4.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0218971, upper bound: 0.0219437
time: 3.19 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 8.70 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 8.70
Output dim: 6, lower bound: -0.0216765, upper bound: 0.0216694
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 8.70
Output dim: 6, lower bound: -0.0216765, upper bound: 0.0216694
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 8.70
Output dim: 6, lower bound: -0.0219345, upper bound: 0.0218989
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 8.70
Output dim: 6, lower bound: -0.0218971, upper bound: 0.0219437

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0361285, 0.0360867
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 75

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0212640, upper bound: 0.0212616
time: 3.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0212640, upper bound: 0.0212620
time: 3.31 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0360871, 0.0361281
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 173

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 219

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213372, upper bound: 0.0213297
time: 3.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213372, upper bound: 0.0213297
time: 3.51 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0361383, 0.0361312
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0195518, upper bound: 0.0195405
time: 1.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0195518, upper bound: 0.0195405
time: 1.74 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0361308, 0.0361389
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0218893, upper bound: 0.0219360
time: 3.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0218894, upper bound: 0.0219357
time: 3.03 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 7.33 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 7.33
Output dim: 6, lower bound: -0.0212640, upper bound: 0.0212616
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 7.33
Output dim: 6, lower bound: -0.0212640, upper bound: 0.0212620
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 7.33
Output dim: 6, lower bound: -0.0213372, upper bound: 0.0213297
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 7.33
Output dim: 6, lower bound: -0.0213372, upper bound: 0.0213297
DS_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 7.33
Output dim: 6, lower bound: -0.0195518, upper bound: 0.0195405
DS_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 7.33
Output dim: 6, lower bound: -0.0195518, upper bound: 0.0195405
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 7.33
Output dim: 6, lower bound: -0.0218893, upper bound: 0.0219360
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 7.33
Output dim: 6, lower bound: -0.0218894, upper bound: 0.0219357

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0360365, 0.0359073
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0207918, upper bound: 0.0207853
time: 2.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0207918, upper bound: 0.0207853
time: 2.81 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0359491, 0.0359933
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 90

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0211902, upper bound: 0.0211781
time: 4.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0211828, upper bound: 0.0211882
time: 2.84 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0357898, 0.0358750
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0206581, upper bound: 0.0206552
time: 3.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0206581, upper bound: 0.0206552
time: 3.34 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0358332, 0.0358307
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 165

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213210, upper bound: 0.0213124
time: 3.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213183, upper bound: 0.0213140
time: 3.92 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0361277, 0.0361367
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0218702, upper bound: 0.0219167
time: 3.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0218702, upper bound: 0.0219167
time: 4.34 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0361286, 0.0361359
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0218824, upper bound: 0.0219286
time: 3.02 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0218823, upper bound: 0.0219287
time: 3.71 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 7.46 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 7.46
Output dim: 6, lower bound: -0.0207918, upper bound: 0.0207853
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 7.46
Output dim: 6, lower bound: -0.0207918, upper bound: 0.0207853
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 7.46
Output dim: 6, lower bound: -0.0211902, upper bound: 0.0211781
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 7.46
Output dim: 6, lower bound: -0.0211828, upper bound: 0.0211882
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 7.46
Output dim: 6, lower bound: -0.0206581, upper bound: 0.0206552
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 7.46
Output dim: 6, lower bound: -0.0206581, upper bound: 0.0206552
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 7.46
Output dim: 6, lower bound: -0.0213210, upper bound: 0.0213124
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 7.46
Output dim: 6, lower bound: -0.0213183, upper bound: 0.0213140
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 7.46
Output dim: 6, lower bound: -0.0218702, upper bound: 0.0219167
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 7.46
Output dim: 6, lower bound: -0.0218702, upper bound: 0.0219167
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 7.46
Output dim: 6, lower bound: -0.0218824, upper bound: 0.0219286
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 7.46
Output dim: 6, lower bound: -0.0218823, upper bound: 0.0219287

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0360049, 0.0358583
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 199

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0206974, upper bound: 0.0206801
time: 3.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0206833, upper bound: 0.0206903
time: 3.13 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0360365, 0.0358757
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 165

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0207843, upper bound: 0.0207775
time: 3.12 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0207842, upper bound: 0.0207775
time: 3.08 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0358807, 0.0359048
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0211698, upper bound: 0.0211578
time: 3.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0211685, upper bound: 0.0211583
time: 2.76 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0358551, 0.0359249
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 173

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0211294, upper bound: 0.0211064
time: 3.36 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0211001, upper bound: 0.0211347
time: 3.37 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0358195, 0.0358123
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 240

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213194, upper bound: 0.0213101
time: 3.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213167, upper bound: 0.0213114
time: 3.47 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0358147, 0.0358184
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 240

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213166, upper bound: 0.0213109
time: 3.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213151, upper bound: 0.0213127
time: 3.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0361044, 0.0361306
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0210663, upper bound: 0.0210958
time: 2.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0210663, upper bound: 0.0210958
time: 2.74 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0361220, 0.0361134
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 75

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0215415, upper bound: 0.0215664
time: 3.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0215391, upper bound: 0.0215736
time: 3.51 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0359053, 0.0359114
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 198

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0217668, upper bound: 0.0217097
time: 3.92 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0216861, upper bound: 0.0218170
time: 3.38 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0359041, 0.0359127
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0215348, upper bound: 0.0215625
time: 3.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0215348, upper bound: 0.0215625
time: 3.33 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 7.67 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 7.67
Output dim: 6, lower bound: -0.0206974, upper bound: 0.0206801
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 7.67
Output dim: 6, lower bound: -0.0206833, upper bound: 0.0206903
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 7.67
Output dim: 6, lower bound: -0.0207843, upper bound: 0.0207775
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 7.67
Output dim: 6, lower bound: -0.0207842, upper bound: 0.0207775
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 7.67
Output dim: 6, lower bound: -0.0211698, upper bound: 0.0211578
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 7.67
Output dim: 6, lower bound: -0.0211685, upper bound: 0.0211583
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 7.67
Output dim: 6, lower bound: -0.0211294, upper bound: 0.0211064
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 7.67
Output dim: 6, lower bound: -0.0211001, upper bound: 0.0211347
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 7.67
Output dim: 6, lower bound: -0.0213194, upper bound: 0.0213101
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 7.67
Output dim: 6, lower bound: -0.0213167, upper bound: 0.0213114
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 7.67
Output dim: 6, lower bound: -0.0213166, upper bound: 0.0213109
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 7.67
Output dim: 6, lower bound: -0.0213151, upper bound: 0.0213127
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 7.67
Output dim: 6, lower bound: -0.0210663, upper bound: 0.0210958
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 7.67
Output dim: 6, lower bound: -0.0210663, upper bound: 0.0210958
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 7.67
Output dim: 6, lower bound: -0.0215415, upper bound: 0.0215664
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 7.67
Output dim: 6, lower bound: -0.0215391, upper bound: 0.0215736
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 7.67
Output dim: 6, lower bound: -0.0217668, upper bound: 0.0217097
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 7.67
Output dim: 6, lower bound: -0.0216861, upper bound: 0.0218170
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 7.67
Output dim: 6, lower bound: -0.0215348, upper bound: 0.0215625
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 7.67
Output dim: 6, lower bound: -0.0215348, upper bound: 0.0215625

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0358124, 0.0356497
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 165

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 199

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0206899, upper bound: 0.0206718
time: 2.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0206755, upper bound: 0.0206826
time: 2.71 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0358123, 0.0356510
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 193

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 219

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0203301, upper bound: 0.0203270
time: 2.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0203301, upper bound: 0.0203270
time: 2.70 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0358595, 0.0359144
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 130

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 77

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0203662, upper bound: 0.0203527
time: 2.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0203662, upper bound: 0.0203527
time: 2.30 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0358737, 0.0358837
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 173

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0211151, upper bound: 0.0210758
time: 4.13 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0210894, upper bound: 0.0211050
time: 10.99 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0356212, 0.0356276
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0211092, upper bound: 0.0210865
time: 3.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0211084, upper bound: 0.0210868
time: 3.09 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0355578, 0.0356941
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 220

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0203499, upper bound: 0.0203785
time: 2.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0203499, upper bound: 0.0203785
time: 2.78 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0357007, 0.0356780
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0211974, upper bound: 0.0211287
time: 3.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0211350, upper bound: 0.0211879
time: 9.06 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0356952, 0.0356935
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 195

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0212735, upper bound: 0.0211725
time: 3.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0211700, upper bound: 0.0212680
time: 3.75 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0356959, 0.0356830
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0212862, upper bound: 0.0212800
time: 4.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0212838, upper bound: 0.0212806
time: 3.52 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0356904, 0.0356996
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0212849, upper bound: 0.0212813
time: 3.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0212826, upper bound: 0.0212823
time: 3.72 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0360922, 0.0361170
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0210360, upper bound: 0.0210644
time: 3.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0210363, upper bound: 0.0210653
time: 3.18 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0361044, 0.0361183
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 195

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0210246, upper bound: 0.0209561
time: 3.90 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0209417, upper bound: 0.0210540
time: 3.62 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0360276, 0.0359258
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0215416, upper bound: 0.0215513
time: 3.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0215285, upper bound: 0.0215667
time: 3.38 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0359344, 0.0360133
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0210447, upper bound: 0.0210649
time: 3.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0210447, upper bound: 0.0210650
time: 3.16 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0352396, 0.0351385
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 251

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0216004, upper bound: 0.0215417
time: 4.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0216004, upper bound: 0.0215417
time: 4.04 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0351324, 0.0352552
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0216861, upper bound: 0.0218039
time: 3.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0216790, upper bound: 0.0218170
time: 2.87 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0358637, 0.0358267
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0205262, upper bound: 0.0205441
time: 2.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0205262, upper bound: 0.0205441
time: 2.58 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0358181, 0.0358688
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0192584, upper bound: 0.0192711
time: 2.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0192584, upper bound: 0.0192711
time: 2.39 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 7.39 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 7.39
Output dim: 6, lower bound: -0.0206899, upper bound: 0.0206718
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 7.39
Output dim: 6, lower bound: -0.0206755, upper bound: 0.0206826
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 7.39
Output dim: 6, lower bound: -0.0203301, upper bound: 0.0203270
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 7.39
Output dim: 6, lower bound: -0.0203301, upper bound: 0.0203270
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 7.39
Output dim: 6, lower bound: -0.0203662, upper bound: 0.0203527
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 7.39
Output dim: 6, lower bound: -0.0203662, upper bound: 0.0203527
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 7.39
Output dim: 6, lower bound: -0.0211151, upper bound: 0.0210758
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 7.39
Output dim: 6, lower bound: -0.0210894, upper bound: 0.0211050
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 7.39
Output dim: 6, lower bound: -0.0211092, upper bound: 0.0210865
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 7.39
Output dim: 6, lower bound: -0.0211084, upper bound: 0.0210868
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 7.39
Output dim: 6, lower bound: -0.0203499, upper bound: 0.0203785
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 7.39
Output dim: 6, lower bound: -0.0203499, upper bound: 0.0203785
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 7.39
Output dim: 6, lower bound: -0.0211974, upper bound: 0.0211287
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 7.39
Output dim: 6, lower bound: -0.0211350, upper bound: 0.0211879
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 7.39
Output dim: 6, lower bound: -0.0212735, upper bound: 0.0211725
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 7.39
Output dim: 6, lower bound: -0.0211700, upper bound: 0.0212680
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 7.39
Output dim: 6, lower bound: -0.0212862, upper bound: 0.0212800
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 7.39
Output dim: 6, lower bound: -0.0212838, upper bound: 0.0212806
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 7.39
Output dim: 6, lower bound: -0.0212849, upper bound: 0.0212813
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 7.39
Output dim: 6, lower bound: -0.0212826, upper bound: 0.0212823
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 7.39
Output dim: 6, lower bound: -0.0210360, upper bound: 0.0210644
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 7.39
Output dim: 6, lower bound: -0.0210363, upper bound: 0.0210653
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 7.39
Output dim: 6, lower bound: -0.0210246, upper bound: 0.0209561
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 7.39
Output dim: 6, lower bound: -0.0209417, upper bound: 0.0210540
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 7.39
Output dim: 6, lower bound: -0.0215416, upper bound: 0.0215513
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 7.39
Output dim: 6, lower bound: -0.0215285, upper bound: 0.0215667
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 7.39
Output dim: 6, lower bound: -0.0210447, upper bound: 0.0210649
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 7.39
Output dim: 6, lower bound: -0.0210447, upper bound: 0.0210650
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 7.39
Output dim: 6, lower bound: -0.0216004, upper bound: 0.0215417
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 7.39
Output dim: 6, lower bound: -0.0216004, upper bound: 0.0215417
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 7.39
Output dim: 6, lower bound: -0.0216861, upper bound: 0.0218039
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 7.39
Output dim: 6, lower bound: -0.0216790, upper bound: 0.0218170
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 7.39
Output dim: 6, lower bound: -0.0205262, upper bound: 0.0205441
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 7.39
Output dim: 6, lower bound: -0.0205262, upper bound: 0.0205441
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 7.39
Output dim: 6, lower bound: -0.0192584, upper bound: 0.0192711
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 7.39
Output dim: 6, lower bound: -0.0192584, upper bound: 0.0192711

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0356221, 0.0355669
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0210836, upper bound: 0.0210444
time: 2.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0210833, upper bound: 0.0210440
time: 3.52 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0355569, 0.0356331
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 165

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0209683, upper bound: 0.0209341
time: 2.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0209148, upper bound: 0.0209825
time: 3.43 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0355806, 0.0356116
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0210837, upper bound: 0.0210654
time: 3.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0210874, upper bound: 0.0210645
time: 3.50 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0356006, 0.0355869
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 219

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0207810, upper bound: 0.0207519
time: 3.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0207810, upper bound: 0.0207515
time: 3.51 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0353967, 0.0352054
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0205156, upper bound: 0.0204563
time: 4.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0205156, upper bound: 0.0204563
time: 4.04 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0352281, 0.0353746
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 210

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0209820, upper bound: 0.0210356
time: 4.11 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0209820, upper bound: 0.0210356
time: 4.28 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0349852, 0.0347684
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 173

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0212198, upper bound: 0.0210950
time: 3.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0211785, upper bound: 0.0211182
time: 3.61 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0347701, 0.0349806
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 90

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0204079, upper bound: 0.0204807
time: 3.36 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0204079, upper bound: 0.0204806
time: 3.31 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0355701, 0.0355554
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 153

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 193

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0197276, upper bound: 0.0197257
time: 2.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0197276, upper bound: 0.0197257
time: 2.68 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0355778, 0.0355572
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 199

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0211968, upper bound: 0.0211787
time: 3.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0211785, upper bound: 0.0211927
time: 3.77 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0355645, 0.0355680
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 220

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0206044, upper bound: 0.0206027
time: 3.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0206044, upper bound: 0.0206027
time: 3.41 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0355696, 0.0355737
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 199

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0211954, upper bound: 0.0211785
time: 3.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0211785, upper bound: 0.0211953
time: 3.49 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0359819, 0.0359948
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0210360, upper bound: 0.0210447
time: 3.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0210197, upper bound: 0.0210647
time: 3.88 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0359754, 0.0360068
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 90

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0210282, upper bound: 0.0210580
time: 3.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0210289, upper bound: 0.0210560
time: 3.26 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0353316, 0.0351298
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 130

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 198

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0208855, upper bound: 0.0207451
time: 3.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0208021, upper bound: 0.0208205
time: 3.06 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0351162, 0.0353368
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 165

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0202966, upper bound: 0.0204178
time: 2.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0202966, upper bound: 0.0204178
time: 3.01 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0361596, 0.0360180
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 220

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0205617, upper bound: 0.0205605
time: 2.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0205617, upper bound: 0.0205605
time: 2.36 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0361391, 0.0360579
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0209887, upper bound: 0.0210382
time: 3.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0209887, upper bound: 0.0210382
time: 3.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0359059, 0.0359741
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0203292, upper bound: 0.0203377
time: 3.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0203292, upper bound: 0.0203377
time: 3.63 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0358952, 0.0359864
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 198

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0209208, upper bound: 0.0208553
time: 3.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0208444, upper bound: 0.0209403
time: 3.16 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0348903, 0.0348310
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 90

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0215576, upper bound: 0.0214478
time: 4.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0214784, upper bound: 0.0214991
time: 4.12 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0349152, 0.0347892
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0215576, upper bound: 0.0214478
time: 4.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0214784, upper bound: 0.0214991
time: 4.03 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0352363, 0.0353146
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 240

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0216841, upper bound: 0.0217807
time: 3.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0216809, upper bound: 0.0217951
time: 3.04 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0351951, 0.0353592
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 78

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0214554, upper bound: 0.0215770
time: 4.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0214509, upper bound: 0.0216031
time: 3.93 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 9.25 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 9.25
Output dim: 6, lower bound: -0.0210836, upper bound: 0.0210444
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 9.25
Output dim: 6, lower bound: -0.0210833, upper bound: 0.0210440
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 9.25
Output dim: 6, lower bound: -0.0209683, upper bound: 0.0209341
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 9.25
Output dim: 6, lower bound: -0.0209148, upper bound: 0.0209825
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 9.25
Output dim: 6, lower bound: -0.0210837, upper bound: 0.0210654
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 9.25
Output dim: 6, lower bound: -0.0210874, upper bound: 0.0210645
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 9.25
Output dim: 6, lower bound: -0.0207810, upper bound: 0.0207519
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 9.25
Output dim: 6, lower bound: -0.0207810, upper bound: 0.0207515
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 9.25
Output dim: 6, lower bound: -0.0205156, upper bound: 0.0204563
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 9.25
Output dim: 6, lower bound: -0.0205156, upper bound: 0.0204563
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 9.25
Output dim: 6, lower bound: -0.0209820, upper bound: 0.0210356
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 9.25
Output dim: 6, lower bound: -0.0209820, upper bound: 0.0210356
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 9.25
Output dim: 6, lower bound: -0.0212198, upper bound: 0.0210950
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 9.25
Output dim: 6, lower bound: -0.0211785, upper bound: 0.0211182
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 9.25
Output dim: 6, lower bound: -0.0204079, upper bound: 0.0204807
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 9.25
Output dim: 6, lower bound: -0.0204079, upper bound: 0.0204806
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 9.25
Output dim: 6, lower bound: -0.0197276, upper bound: 0.0197257
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 9.25
Output dim: 6, lower bound: -0.0197276, upper bound: 0.0197257
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 9.25
Output dim: 6, lower bound: -0.0211968, upper bound: 0.0211787
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 9.25
Output dim: 6, lower bound: -0.0211785, upper bound: 0.0211927
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 9.25
Output dim: 6, lower bound: -0.0206044, upper bound: 0.0206027
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 9.25
Output dim: 6, lower bound: -0.0206044, upper bound: 0.0206027
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 9.25
Output dim: 6, lower bound: -0.0211954, upper bound: 0.0211785
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 9.25
Output dim: 6, lower bound: -0.0211785, upper bound: 0.0211953
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 9.25
Output dim: 6, lower bound: -0.0210360, upper bound: 0.0210447
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 9.25
Output dim: 6, lower bound: -0.0210197, upper bound: 0.0210647
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 9.25
Output dim: 6, lower bound: -0.0210282, upper bound: 0.0210580
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 9.25
Output dim: 6, lower bound: -0.0210289, upper bound: 0.0210560
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 9.25
Output dim: 6, lower bound: -0.0208855, upper bound: 0.0207451
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 9.25
Output dim: 6, lower bound: -0.0208021, upper bound: 0.0208205
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 9.25
Output dim: 6, lower bound: -0.0202966, upper bound: 0.0204178
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 9.25
Output dim: 6, lower bound: -0.0202966, upper bound: 0.0204178
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 9.25
Output dim: 6, lower bound: -0.0205617, upper bound: 0.0205605
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 9.25
Output dim: 6, lower bound: -0.0205617, upper bound: 0.0205605
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 9.25
Output dim: 6, lower bound: -0.0209887, upper bound: 0.0210382
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 9.25
Output dim: 6, lower bound: -0.0209887, upper bound: 0.0210382
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 9.25
Output dim: 6, lower bound: -0.0203292, upper bound: 0.0203377
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 9.25
Output dim: 6, lower bound: -0.0203292, upper bound: 0.0203377
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 9.25
Output dim: 6, lower bound: -0.0209208, upper bound: 0.0208553
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 9.25
Output dim: 6, lower bound: -0.0208444, upper bound: 0.0209403
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 9.25
Output dim: 6, lower bound: -0.0215576, upper bound: 0.0214478
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 9.25
Output dim: 6, lower bound: -0.0214784, upper bound: 0.0214991
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 9.25
Output dim: 6, lower bound: -0.0215576, upper bound: 0.0214478
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 9.25
Output dim: 6, lower bound: -0.0214784, upper bound: 0.0214991
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 9.25
Output dim: 6, lower bound: -0.0216841, upper bound: 0.0217807
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 9.25
Output dim: 6, lower bound: -0.0216809, upper bound: 0.0217951
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 9.25
Output dim: 6, lower bound: -0.0214554, upper bound: 0.0215770
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 9.25
Output dim: 6, lower bound: -0.0214509, upper bound: 0.0216031

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0355292, 0.0354586
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0210755, upper bound: 0.0210365
time: 3.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0210749, upper bound: 0.0210368
time: 2.97 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0355214, 0.0354740
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 90

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0205348, upper bound: 0.0204987
time: 2.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0205348, upper bound: 0.0204987
time: 2.88 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0353148, 0.0352180
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0209372, upper bound: 0.0209036
time: 3.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0209372, upper bound: 0.0209007
time: 3.13 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0351418, 0.0354004
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 90

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0203791, upper bound: 0.0204374
time: 3.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0203791, upper bound: 0.0204374
time: 3.17 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0354259, 0.0354700
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 240

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0210825, upper bound: 0.0210639
time: 3.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0210822, upper bound: 0.0210642
time: 3.09 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0354470, 0.0354569
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0210558, upper bound: 0.0210263
time: 3.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0210535, upper bound: 0.0210328
time: 3.18 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0352940, 0.0353075
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 193

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0187876, upper bound: 0.0187848
time: 1.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0187876, upper bound: 0.0187848
time: 1.58 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0353497, 0.0352804
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 198

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0206418, upper bound: 0.0205336
time: 3.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0205536, upper bound: 0.0206121
time: 3.87 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0352129, 0.0353597
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 173

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0209268, upper bound: 0.0209407
time: 3.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0208928, upper bound: 0.0209808
time: 3.13 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0352132, 0.0353588
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 78

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0207678, upper bound: 0.0208093
time: 3.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0207603, upper bound: 0.0208208
time: 3.29 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 7.74 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 7.74
Output dim: 6, lower bound: -0.0210755, upper bound: 0.0210365
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 7.74
Output dim: 6, lower bound: -0.0210749, upper bound: 0.0210368
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 7.74
Output dim: 6, lower bound: -0.0205348, upper bound: 0.0204987
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 7.74
Output dim: 6, lower bound: -0.0205348, upper bound: 0.0204987
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 7.74
Output dim: 6, lower bound: -0.0209372, upper bound: 0.0209036
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 7.74
Output dim: 6, lower bound: -0.0209372, upper bound: 0.0209007
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 7.74
Output dim: 6, lower bound: -0.0203791, upper bound: 0.0204374
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 7.74
Output dim: 6, lower bound: -0.0203791, upper bound: 0.0204374
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 7.74
Output dim: 6, lower bound: -0.0210825, upper bound: 0.0210639
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 7.74
Output dim: 6, lower bound: -0.0210822, upper bound: 0.0210642
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 7.74
Output dim: 6, lower bound: -0.0210558, upper bound: 0.0210263
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 7.74
Output dim: 6, lower bound: -0.0210535, upper bound: 0.0210328
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 7.74
Output dim: 6, lower bound: -0.0187876, upper bound: 0.0187848
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 7.74
Output dim: 6, lower bound: -0.0187876, upper bound: 0.0187848
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 7.74
Output dim: 6, lower bound: -0.0206418, upper bound: 0.0205336
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 7.74
Output dim: 6, lower bound: -0.0205536, upper bound: 0.0206121
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 7.74
Output dim: 6, lower bound: -0.0209268, upper bound: 0.0209407
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 7.74
Output dim: 6, lower bound: -0.0208928, upper bound: 0.0209808
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 7.74
Output dim: 6, lower bound: -0.0207678, upper bound: 0.0208093
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 7.74
Output dim: 6, lower bound: -0.0207603, upper bound: 0.0208208
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 7.74
Output dim: 6, lower bound: -0.0212198, upper bound: 0.0210950
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 7.74
Output dim: 6, lower bound: -0.0211785, upper bound: 0.0211182
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 7.74
Output dim: 6, lower bound: -0.0211968, upper bound: 0.0211787
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 7.74
Output dim: 6, lower bound: -0.0211785, upper bound: 0.0211927
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 7.74
Output dim: 6, lower bound: -0.0211954, upper bound: 0.0211785
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 7.74
Output dim: 6, lower bound: -0.0211785, upper bound: 0.0211953
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 7.74
Output dim: 6, lower bound: -0.0210360, upper bound: 0.0210447
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 7.74
Output dim: 6, lower bound: -0.0210197, upper bound: 0.0210647
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 7.74
Output dim: 6, lower bound: -0.0210282, upper bound: 0.0210580
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 7.74
Output dim: 6, lower bound: -0.0210289, upper bound: 0.0210560
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 7.74
Output dim: 6, lower bound: -0.0208855, upper bound: 0.0207451
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 7.74
Output dim: 6, lower bound: -0.0208021, upper bound: 0.0208205
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 7.74
Output dim: 6, lower bound: -0.0209887, upper bound: 0.0210382
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 7.74
Output dim: 6, lower bound: -0.0209887, upper bound: 0.0210382
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 7.74
Output dim: 6, lower bound: -0.0209208, upper bound: 0.0208553
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 7.74
Output dim: 6, lower bound: -0.0208444, upper bound: 0.0209403
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 7.74
Output dim: 6, lower bound: -0.0215576, upper bound: 0.0214478
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 7.74
Output dim: 6, lower bound: -0.0214784, upper bound: 0.0214991
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 7.74
Output dim: 6, lower bound: -0.0215576, upper bound: 0.0214478
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 7.74
Output dim: 6, lower bound: -0.0214784, upper bound: 0.0214991
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 7.74
Output dim: 6, lower bound: -0.0216841, upper bound: 0.0217807
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 7.74
Output dim: 6, lower bound: -0.0216809, upper bound: 0.0217951
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 7.74
Output dim: 6, lower bound: -0.0214554, upper bound: 0.0215770
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 7.74
Output dim: 6, lower bound: -0.0214509, upper bound: 0.0216031

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 5.85 + 595.91 = 601.77 seconds
