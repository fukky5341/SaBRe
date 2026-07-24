## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.020780391999999998


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

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
execution time: IAR + RelationalAnalysis = 1.33 + 5.28 = 6.61 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.0221068, upper bound: 0.0221068

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0220997, upper bound: 0.0220997
time: 3.99 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0220997, upper bound: 0.0220997
time: 4.07 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 8.21 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 8.21
Output dim: 6, lower bound: -0.0220997, upper bound: 0.0220997
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 8.21
Output dim: 6, lower bound: -0.0220997, upper bound: 0.0220997

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0220676, upper bound: 0.0220667
time: 3.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0220667, upper bound: 0.0220676
time: 3.32 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0220676, upper bound: 0.0220667
time: 3.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0220667, upper bound: 0.0220676
time: 4.01 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 9.19 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 9.19
Output dim: 6, lower bound: -0.0220676, upper bound: 0.0220667
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 9.19
Output dim: 6, lower bound: -0.0220667, upper bound: 0.0220676
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 9.19
Output dim: 6, lower bound: -0.0220676, upper bound: 0.0220667
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 9.19
Output dim: 6, lower bound: -0.0220667, upper bound: 0.0220676

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0215573, upper bound: 0.0215573
time: 2.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0215573, upper bound: 0.0215573
time: 2.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0215569, upper bound: 0.0215575
time: 3.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0215569, upper bound: 0.0215575
time: 4.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0215575, upper bound: 0.0215569
time: 3.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0215575, upper bound: 0.0215569
time: 3.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0215573, upper bound: 0.0215573
time: 3.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0215573, upper bound: 0.0215573
time: 3.35 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 7.84 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 7.84
Output dim: 6, lower bound: -0.0215573, upper bound: 0.0215573
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 7.84
Output dim: 6, lower bound: -0.0215573, upper bound: 0.0215573
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 7.84
Output dim: 6, lower bound: -0.0215569, upper bound: 0.0215575
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 7.84
Output dim: 6, lower bound: -0.0215569, upper bound: 0.0215575
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 7.84
Output dim: 6, lower bound: -0.0215575, upper bound: 0.0215569
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 7.84
Output dim: 6, lower bound: -0.0215575, upper bound: 0.0215569
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 7.84
Output dim: 6, lower bound: -0.0215573, upper bound: 0.0215573
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 7.84
Output dim: 6, lower bound: -0.0215573, upper bound: 0.0215573

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0215389, upper bound: 0.0215390
time: 3.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0215388, upper bound: 0.0215391
time: 4.08 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0215389, upper bound: 0.0215391
time: 3.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0215388, upper bound: 0.0215390
time: 3.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0215386, upper bound: 0.0215392
time: 3.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0215386, upper bound: 0.0215392
time: 3.44 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0215386, upper bound: 0.0215391
time: 4.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0215386, upper bound: 0.0215392
time: 3.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0215392, upper bound: 0.0215386
time: 3.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0215392, upper bound: 0.0215386
time: 3.24 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0215392, upper bound: 0.0215386
time: 3.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0215392, upper bound: 0.0215386
time: 3.46 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0215391, upper bound: 0.0215389
time: 3.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0215391, upper bound: 0.0215389
time: 3.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0215391, upper bound: 0.0215389
time: 3.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0215391, upper bound: 0.0215389
time: 4.32 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 9.20 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 9.20
Output dim: 6, lower bound: -0.0215389, upper bound: 0.0215390
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 9.20
Output dim: 6, lower bound: -0.0215388, upper bound: 0.0215391
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 9.20
Output dim: 6, lower bound: -0.0215389, upper bound: 0.0215391
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 9.20
Output dim: 6, lower bound: -0.0215388, upper bound: 0.0215390
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 9.20
Output dim: 6, lower bound: -0.0215386, upper bound: 0.0215392
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 9.20
Output dim: 6, lower bound: -0.0215386, upper bound: 0.0215392
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 9.20
Output dim: 6, lower bound: -0.0215386, upper bound: 0.0215391
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 9.20
Output dim: 6, lower bound: -0.0215386, upper bound: 0.0215392
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 9.20
Output dim: 6, lower bound: -0.0215392, upper bound: 0.0215386
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 9.20
Output dim: 6, lower bound: -0.0215392, upper bound: 0.0215386
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 9.20
Output dim: 6, lower bound: -0.0215392, upper bound: 0.0215386
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 9.20
Output dim: 6, lower bound: -0.0215392, upper bound: 0.0215386
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 9.20
Output dim: 6, lower bound: -0.0215391, upper bound: 0.0215389
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 9.20
Output dim: 6, lower bound: -0.0215391, upper bound: 0.0215389
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 9.20
Output dim: 6, lower bound: -0.0215391, upper bound: 0.0215389
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 9.20
Output dim: 6, lower bound: -0.0215391, upper bound: 0.0215389

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0214380, upper bound: 0.0213767
time: 3.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213687, upper bound: 0.0214382
time: 3.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0214379, upper bound: 0.0213771
time: 3.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213687, upper bound: 0.0214383
time: 3.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0214380, upper bound: 0.0213767
time: 3.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213687, upper bound: 0.0214382
time: 3.41 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0214379, upper bound: 0.0213771
time: 3.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213687, upper bound: 0.0214382
time: 3.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0214376, upper bound: 0.0213761
time: 3.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213692, upper bound: 0.0214384
time: 3.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0214376, upper bound: 0.0213763
time: 3.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213692, upper bound: 0.0214384
time: 7.97 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0214376, upper bound: 0.0213761
time: 3.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213692, upper bound: 0.0214384
time: 3.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0214376, upper bound: 0.0213763
time: 4.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213692, upper bound: 0.0214384
time: 8.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0214384, upper bound: 0.0213692
time: 3.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213763, upper bound: 0.0214376
time: 2.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0214384, upper bound: 0.0213692
time: 4.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213761, upper bound: 0.0214376
time: 3.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0214384, upper bound: 0.0213692
time: 3.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213763, upper bound: 0.0214376
time: 2.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0214384, upper bound: 0.0213692
time: 3.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213761, upper bound: 0.0214376
time: 3.48 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0214382, upper bound: 0.0213687
time: 3.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213771, upper bound: 0.0214379
time: 3.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0214382, upper bound: 0.0213687
time: 3.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213767, upper bound: 0.0214380
time: 3.07 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0214382, upper bound: 0.0213687
time: 3.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213771, upper bound: 0.0214379
time: 3.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0214382, upper bound: 0.0213687
time: 3.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213767, upper bound: 0.0214380
time: 3.01 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 7.80 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.80
Output dim: 6, lower bound: -0.0214380, upper bound: 0.0213767
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.80
Output dim: 6, lower bound: -0.0213687, upper bound: 0.0214382
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.80
Output dim: 6, lower bound: -0.0214379, upper bound: 0.0213771
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.80
Output dim: 6, lower bound: -0.0213687, upper bound: 0.0214383
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.80
Output dim: 6, lower bound: -0.0214380, upper bound: 0.0213767
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.80
Output dim: 6, lower bound: -0.0213687, upper bound: 0.0214382
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.80
Output dim: 6, lower bound: -0.0214379, upper bound: 0.0213771
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.80
Output dim: 6, lower bound: -0.0213687, upper bound: 0.0214382
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.80
Output dim: 6, lower bound: -0.0214376, upper bound: 0.0213761
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.80
Output dim: 6, lower bound: -0.0213692, upper bound: 0.0214384
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.80
Output dim: 6, lower bound: -0.0214376, upper bound: 0.0213763
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.80
Output dim: 6, lower bound: -0.0213692, upper bound: 0.0214384
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.80
Output dim: 6, lower bound: -0.0214376, upper bound: 0.0213761
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.80
Output dim: 6, lower bound: -0.0213692, upper bound: 0.0214384
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.80
Output dim: 6, lower bound: -0.0214376, upper bound: 0.0213763
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.80
Output dim: 6, lower bound: -0.0213692, upper bound: 0.0214384
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.80
Output dim: 6, lower bound: -0.0214384, upper bound: 0.0213692
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.80
Output dim: 6, lower bound: -0.0213763, upper bound: 0.0214376
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.80
Output dim: 6, lower bound: -0.0214384, upper bound: 0.0213692
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.80
Output dim: 6, lower bound: -0.0213761, upper bound: 0.0214376
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.80
Output dim: 6, lower bound: -0.0214384, upper bound: 0.0213692
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.80
Output dim: 6, lower bound: -0.0213763, upper bound: 0.0214376
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.80
Output dim: 6, lower bound: -0.0214384, upper bound: 0.0213692
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.80
Output dim: 6, lower bound: -0.0213761, upper bound: 0.0214376
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.80
Output dim: 6, lower bound: -0.0214382, upper bound: 0.0213687
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.80
Output dim: 6, lower bound: -0.0213771, upper bound: 0.0214379
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.80
Output dim: 6, lower bound: -0.0214382, upper bound: 0.0213687
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.80
Output dim: 6, lower bound: -0.0213767, upper bound: 0.0214380
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.80
Output dim: 6, lower bound: -0.0214382, upper bound: 0.0213687
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.80
Output dim: 6, lower bound: -0.0213771, upper bound: 0.0214379
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.80
Output dim: 6, lower bound: -0.0214382, upper bound: 0.0213687
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.80
Output dim: 6, lower bound: -0.0213767, upper bound: 0.0214380

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213556, upper bound: 0.0212890
time: 3.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213478, upper bound: 0.0212937
time: 4.10 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0212858, upper bound: 0.0213482
time: 3.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0212818, upper bound: 0.0213559
time: 3.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213555, upper bound: 0.0212890
time: 3.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213477, upper bound: 0.0212943
time: 3.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0212858, upper bound: 0.0213482
time: 3.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0212819, upper bound: 0.0213560
time: 3.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213556, upper bound: 0.0212890
time: 3.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213478, upper bound: 0.0212938
time: 3.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0212858, upper bound: 0.0213482
time: 3.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0212818, upper bound: 0.0213559
time: 3.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213555, upper bound: 0.0212890
time: 3.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213477, upper bound: 0.0212943
time: 3.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0212858, upper bound: 0.0213482
time: 3.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0212819, upper bound: 0.0213560
time: 3.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213553, upper bound: 0.0212888
time: 3.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213475, upper bound: 0.0212933
time: 4.23 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0212863, upper bound: 0.0213482
time: 3.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0212822, upper bound: 0.0213561
time: 3.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213553, upper bound: 0.0212888
time: 3.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213476, upper bound: 0.0212936
time: 3.21 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0212863, upper bound: 0.0213482
time: 3.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0212823, upper bound: 0.0213561
time: 3.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213553, upper bound: 0.0212887
time: 3.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213475, upper bound: 0.0212933
time: 3.44 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0212863, upper bound: 0.0213482
time: 3.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0212822, upper bound: 0.0213561
time: 3.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213553, upper bound: 0.0212888
time: 4.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213476, upper bound: 0.0212936
time: 3.23 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0212863, upper bound: 0.0213482
time: 4.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0212823, upper bound: 0.0213561
time: 3.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213561, upper bound: 0.0212823
time: 3.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213482, upper bound: 0.0212863
time: 3.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0212936, upper bound: 0.0213476
time: 4.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0212888, upper bound: 0.0213553
time: 3.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213561, upper bound: 0.0212822
time: 3.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213482, upper bound: 0.0212863
time: 3.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0212933, upper bound: 0.0213475
time: 3.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0212887, upper bound: 0.0213553
time: 3.84 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213561, upper bound: 0.0212823
time: 3.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213482, upper bound: 0.0212863
time: 3.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0212936, upper bound: 0.0213476
time: 4.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0212888, upper bound: 0.0213553
time: 3.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213561, upper bound: 0.0212822
time: 3.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213482, upper bound: 0.0212863
time: 3.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0212933, upper bound: 0.0213475
time: 3.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0212887, upper bound: 0.0213553
time: 3.21 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213560, upper bound: 0.0212819
time: 3.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213482, upper bound: 0.0212858
time: 3.92 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0212943, upper bound: 0.0213477
time: 3.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0212890, upper bound: 0.0213555
time: 3.52 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213559, upper bound: 0.0212818
time: 3.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213482, upper bound: 0.0212858
time: 3.42 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0212937, upper bound: 0.0213477
time: 3.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0212889, upper bound: 0.0213556
time: 3.44 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213560, upper bound: 0.0212819
time: 3.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213482, upper bound: 0.0212858
time: 3.24 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0212943, upper bound: 0.0213477
time: 3.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0212890, upper bound: 0.0213556
time: 3.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213559, upper bound: 0.0212818
time: 9.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213482, upper bound: 0.0212858
time: 4.05 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0212937, upper bound: 0.0213477
time: 3.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0212889, upper bound: 0.0213556
time: 3.20 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 8.52 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.52
Output dim: 6, lower bound: -0.0213556, upper bound: 0.0212890
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.52
Output dim: 6, lower bound: -0.0213478, upper bound: 0.0212937
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.52
Output dim: 6, lower bound: -0.0212858, upper bound: 0.0213482
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.52
Output dim: 6, lower bound: -0.0212818, upper bound: 0.0213559
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.52
Output dim: 6, lower bound: -0.0213555, upper bound: 0.0212890
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.52
Output dim: 6, lower bound: -0.0213477, upper bound: 0.0212943
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.52
Output dim: 6, lower bound: -0.0212858, upper bound: 0.0213482
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.52
Output dim: 6, lower bound: -0.0212819, upper bound: 0.0213560
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.52
Output dim: 6, lower bound: -0.0213556, upper bound: 0.0212890
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.52
Output dim: 6, lower bound: -0.0213478, upper bound: 0.0212938
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.52
Output dim: 6, lower bound: -0.0212858, upper bound: 0.0213482
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.52
Output dim: 6, lower bound: -0.0212818, upper bound: 0.0213559
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.52
Output dim: 6, lower bound: -0.0213555, upper bound: 0.0212890
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.52
Output dim: 6, lower bound: -0.0213477, upper bound: 0.0212943
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.52
Output dim: 6, lower bound: -0.0212858, upper bound: 0.0213482
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.52
Output dim: 6, lower bound: -0.0212819, upper bound: 0.0213560
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.52
Output dim: 6, lower bound: -0.0213553, upper bound: 0.0212888
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.52
Output dim: 6, lower bound: -0.0213475, upper bound: 0.0212933
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.52
Output dim: 6, lower bound: -0.0212863, upper bound: 0.0213482
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.52
Output dim: 6, lower bound: -0.0212822, upper bound: 0.0213561
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.52
Output dim: 6, lower bound: -0.0213553, upper bound: 0.0212888
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.52
Output dim: 6, lower bound: -0.0213476, upper bound: 0.0212936
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.52
Output dim: 6, lower bound: -0.0212863, upper bound: 0.0213482
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.52
Output dim: 6, lower bound: -0.0212823, upper bound: 0.0213561
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.52
Output dim: 6, lower bound: -0.0213553, upper bound: 0.0212887
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.52
Output dim: 6, lower bound: -0.0213475, upper bound: 0.0212933
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.52
Output dim: 6, lower bound: -0.0212863, upper bound: 0.0213482
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.52
Output dim: 6, lower bound: -0.0212822, upper bound: 0.0213561
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.52
Output dim: 6, lower bound: -0.0213553, upper bound: 0.0212888
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.52
Output dim: 6, lower bound: -0.0213476, upper bound: 0.0212936
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.52
Output dim: 6, lower bound: -0.0212863, upper bound: 0.0213482
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.52
Output dim: 6, lower bound: -0.0212823, upper bound: 0.0213561
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.52
Output dim: 6, lower bound: -0.0213561, upper bound: 0.0212823
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.52
Output dim: 6, lower bound: -0.0213482, upper bound: 0.0212863
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.52
Output dim: 6, lower bound: -0.0212936, upper bound: 0.0213476
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.52
Output dim: 6, lower bound: -0.0212888, upper bound: 0.0213553
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.52
Output dim: 6, lower bound: -0.0213561, upper bound: 0.0212822
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.52
Output dim: 6, lower bound: -0.0213482, upper bound: 0.0212863
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.52
Output dim: 6, lower bound: -0.0212933, upper bound: 0.0213475
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.52
Output dim: 6, lower bound: -0.0212887, upper bound: 0.0213553
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.52
Output dim: 6, lower bound: -0.0213561, upper bound: 0.0212823
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.52
Output dim: 6, lower bound: -0.0213482, upper bound: 0.0212863
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.52
Output dim: 6, lower bound: -0.0212936, upper bound: 0.0213476
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.52
Output dim: 6, lower bound: -0.0212888, upper bound: 0.0213553
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.52
Output dim: 6, lower bound: -0.0213561, upper bound: 0.0212822
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.52
Output dim: 6, lower bound: -0.0213482, upper bound: 0.0212863
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.52
Output dim: 6, lower bound: -0.0212933, upper bound: 0.0213475
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.52
Output dim: 6, lower bound: -0.0212887, upper bound: 0.0213553
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.52
Output dim: 6, lower bound: -0.0213560, upper bound: 0.0212819
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.52
Output dim: 6, lower bound: -0.0213482, upper bound: 0.0212858
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.52
Output dim: 6, lower bound: -0.0212943, upper bound: 0.0213477
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.52
Output dim: 6, lower bound: -0.0212890, upper bound: 0.0213555
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.52
Output dim: 6, lower bound: -0.0213559, upper bound: 0.0212818
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.52
Output dim: 6, lower bound: -0.0213482, upper bound: 0.0212858
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.52
Output dim: 6, lower bound: -0.0212937, upper bound: 0.0213477
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.52
Output dim: 6, lower bound: -0.0212889, upper bound: 0.0213556
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.52
Output dim: 6, lower bound: -0.0213560, upper bound: 0.0212819
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.52
Output dim: 6, lower bound: -0.0213482, upper bound: 0.0212858
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.52
Output dim: 6, lower bound: -0.0212943, upper bound: 0.0213477
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.52
Output dim: 6, lower bound: -0.0212890, upper bound: 0.0213556
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.52
Output dim: 6, lower bound: -0.0213559, upper bound: 0.0212818
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.52
Output dim: 6, lower bound: -0.0213482, upper bound: 0.0212858
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.52
Output dim: 6, lower bound: -0.0212937, upper bound: 0.0213477
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.52
Output dim: 6, lower bound: -0.0212889, upper bound: 0.0213556

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0353923, 0.0351824
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0186289, upper bound: 0.0186099
time: 1.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0186289, upper bound: 0.0186099
time: 1.51 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0353717, 0.0352033
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0186262, upper bound: 0.0186141
time: 2.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0186262, upper bound: 0.0186141
time: 2.09 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0352125, 0.0353397
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0186177, upper bound: 0.0186224
time: 1.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0186177, upper bound: 0.0186224
time: 1.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0351911, 0.0353619
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0186122, upper bound: 0.0186249
time: 1.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0186122, upper bound: 0.0186249
time: 1.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0354106, 0.0351779
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0186247, upper bound: 0.0186123
time: 1.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0186247, upper bound: 0.0186123
time: 1.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0353934, 0.0352015
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0186224, upper bound: 0.0186185
time: 2.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0186224, upper bound: 0.0186185
time: 2.02 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0352317, 0.0353268
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0186139, upper bound: 0.0186260
time: 2.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0186139, upper bound: 0.0186260
time: 2.04 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0352133, 0.0353522
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0186097, upper bound: 0.0186289
time: 2.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0186097, upper bound: 0.0186289
time: 2.16 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 5.73 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.73
Output dim: 6, lower bound: -0.0186289, upper bound: 0.0186099
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.73
Output dim: 6, lower bound: -0.0186289, upper bound: 0.0186099
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.73
Output dim: 6, lower bound: -0.0186262, upper bound: 0.0186141
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.73
Output dim: 6, lower bound: -0.0186262, upper bound: 0.0186141
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.73
Output dim: 6, lower bound: -0.0186177, upper bound: 0.0186224
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.73
Output dim: 6, lower bound: -0.0186177, upper bound: 0.0186224
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.73
Output dim: 6, lower bound: -0.0186122, upper bound: 0.0186249
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.73
Output dim: 6, lower bound: -0.0186122, upper bound: 0.0186249
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.73
Output dim: 6, lower bound: -0.0186247, upper bound: 0.0186123
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.73
Output dim: 6, lower bound: -0.0186247, upper bound: 0.0186123
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.73
Output dim: 6, lower bound: -0.0186224, upper bound: 0.0186185
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.73
Output dim: 6, lower bound: -0.0186224, upper bound: 0.0186185
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.73
Output dim: 6, lower bound: -0.0186139, upper bound: 0.0186260
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.73
Output dim: 6, lower bound: -0.0186139, upper bound: 0.0186260
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.73
Output dim: 6, lower bound: -0.0186097, upper bound: 0.0186289
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.73
Output dim: 6, lower bound: -0.0186097, upper bound: 0.0186289
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.0213556, upper bound: 0.0212890
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.0213478, upper bound: 0.0212938
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.0212858, upper bound: 0.0213482
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.0212818, upper bound: 0.0213559
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.0213555, upper bound: 0.0212890
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.0213477, upper bound: 0.0212943
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.0212858, upper bound: 0.0213482
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.0212819, upper bound: 0.0213560
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.0213553, upper bound: 0.0212888
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.0213475, upper bound: 0.0212933
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.0212863, upper bound: 0.0213482
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.0212822, upper bound: 0.0213561
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.0213553, upper bound: 0.0212888
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.0213476, upper bound: 0.0212936
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.0212863, upper bound: 0.0213482
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.0212823, upper bound: 0.0213561
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.0213553, upper bound: 0.0212887
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.0213475, upper bound: 0.0212933
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.0212863, upper bound: 0.0213482
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.0212822, upper bound: 0.0213561
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.0213553, upper bound: 0.0212888
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.0213476, upper bound: 0.0212936
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.0212863, upper bound: 0.0213482
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.0212823, upper bound: 0.0213561
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.0213561, upper bound: 0.0212823
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.0213482, upper bound: 0.0212863
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.0212936, upper bound: 0.0213476
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.0212888, upper bound: 0.0213553
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.0213561, upper bound: 0.0212822
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.0213482, upper bound: 0.0212863
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.0212933, upper bound: 0.0213475
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.0212887, upper bound: 0.0213553
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.0213561, upper bound: 0.0212823
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.0213482, upper bound: 0.0212863
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.0212936, upper bound: 0.0213476
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.0212888, upper bound: 0.0213553
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.0213561, upper bound: 0.0212822
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.0213482, upper bound: 0.0212863
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.0212933, upper bound: 0.0213475
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.0212887, upper bound: 0.0213553
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.0213560, upper bound: 0.0212819
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.0213482, upper bound: 0.0212858
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.0212943, upper bound: 0.0213477
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.0212890, upper bound: 0.0213555
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.0213559, upper bound: 0.0212818
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.0213482, upper bound: 0.0212858
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.0212937, upper bound: 0.0213477
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.0212889, upper bound: 0.0213556
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.0213560, upper bound: 0.0212819
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.0213482, upper bound: 0.0212858
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.0212943, upper bound: 0.0213477
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.0212890, upper bound: 0.0213556
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.0213559, upper bound: 0.0212818
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.0213482, upper bound: 0.0212858
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.0212937, upper bound: 0.0213477
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.73
Output dim: 6, lower bound: -0.0212889, upper bound: 0.0213556

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 6.61 + 597.03 = 603.64 seconds
