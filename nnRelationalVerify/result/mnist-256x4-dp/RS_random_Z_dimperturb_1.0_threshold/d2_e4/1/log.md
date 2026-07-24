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
execution time: IAR + RelationalAnalysis = 1.32 + 5.20 = 6.52 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.0221068, upper bound: 0.0221068

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 90

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0221011, upper bound: 0.0220920
time: 3.24 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0220920, upper bound: 0.0221011
time: 4.02 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 7.27 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 7.27
Output dim: 6, lower bound: -0.0221011, upper bound: 0.0220920
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 7.27
Output dim: 6, lower bound: -0.0220920, upper bound: 0.0221011

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0360737, 0.0360627
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
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0220351, upper bound: 0.0220313
time: 3.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0220351, upper bound: 0.0220313
time: 3.98 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0360627, 0.0360737
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
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0220781, upper bound: 0.0220845
time: 3.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0220766, upper bound: 0.0220881
time: 3.58 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 8.46 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 8.46
Output dim: 6, lower bound: -0.0220351, upper bound: 0.0220313
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 8.46
Output dim: 6, lower bound: -0.0220351, upper bound: 0.0220313
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 8.46
Output dim: 6, lower bound: -0.0220781, upper bound: 0.0220845
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 8.46
Output dim: 6, lower bound: -0.0220766, upper bound: 0.0220881

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0358611, 0.0358470
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
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 153

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0220040, upper bound: 0.0220000
time: 4.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0220037, upper bound: 0.0220004
time: 3.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0358598, 0.0358501
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
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 193

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0220165, upper bound: 0.0220128
time: 4.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0220165, upper bound: 0.0220128
time: 3.91 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0360416, 0.0360393
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
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0220597, upper bound: 0.0220664
time: 4.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0220597, upper bound: 0.0220663
time: 3.52 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0360301, 0.0360526
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 153

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0220331, upper bound: 0.0219590
time: 4.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0219387, upper bound: 0.0220449
time: 4.44 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 9.97 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 9.97
Output dim: 6, lower bound: -0.0220040, upper bound: 0.0220000
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 9.97
Output dim: 6, lower bound: -0.0220037, upper bound: 0.0220004
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 9.97
Output dim: 6, lower bound: -0.0220165, upper bound: 0.0220128
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 9.97
Output dim: 6, lower bound: -0.0220165, upper bound: 0.0220128
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 9.97
Output dim: 6, lower bound: -0.0220597, upper bound: 0.0220664
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 9.97
Output dim: 6, lower bound: -0.0220597, upper bound: 0.0220663
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 9.97
Output dim: 6, lower bound: -0.0220331, upper bound: 0.0219590
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 9.97
Output dim: 6, lower bound: -0.0219387, upper bound: 0.0220449

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0357367, 0.0357239
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213937, upper bound: 0.0213932
time: 3.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213937, upper bound: 0.0213932
time: 5.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0357303, 0.0357227
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0217787, upper bound: 0.0217776
time: 3.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0217787, upper bound: 0.0217776
time: 3.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0358347, 0.0358417
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0218602, upper bound: 0.0218573
time: 4.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0218602, upper bound: 0.0218573
time: 3.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0358451, 0.0358249
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0219025, upper bound: 0.0218638
time: 3.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0218641, upper bound: 0.0218975
time: 5.12 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0360606, 0.0360783
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
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0220526, upper bound: 0.0220590
time: 3.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0220525, upper bound: 0.0220592
time: 3.49 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0360774, 0.0360583
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
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 75

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0215328, upper bound: 0.0215305
time: 3.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0215328, upper bound: 0.0215305
time: 3.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0352854, 0.0351751
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
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0218820, upper bound: 0.0218033
time: 4.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0218820, upper bound: 0.0218033
time: 4.02 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0351526, 0.0352905
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
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0214377, upper bound: 0.0215394
time: 3.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0214377, upper bound: 0.0215394
time: 3.37 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 8.58 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 8.58
Output dim: 6, lower bound: -0.0213937, upper bound: 0.0213932
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 8.58
Output dim: 6, lower bound: -0.0213937, upper bound: 0.0213932
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 8.58
Output dim: 6, lower bound: -0.0217787, upper bound: 0.0217776
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 8.58
Output dim: 6, lower bound: -0.0217787, upper bound: 0.0217776
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 8.58
Output dim: 6, lower bound: -0.0218602, upper bound: 0.0218573
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 8.58
Output dim: 6, lower bound: -0.0218602, upper bound: 0.0218573
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 8.58
Output dim: 6, lower bound: -0.0219025, upper bound: 0.0218638
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 8.58
Output dim: 6, lower bound: -0.0218641, upper bound: 0.0218975
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 8.58
Output dim: 6, lower bound: -0.0220526, upper bound: 0.0220590
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 8.58
Output dim: 6, lower bound: -0.0220525, upper bound: 0.0220592
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 8.58
Output dim: 6, lower bound: -0.0215328, upper bound: 0.0215305
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 8.58
Output dim: 6, lower bound: -0.0215328, upper bound: 0.0215305
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 8.58
Output dim: 6, lower bound: -0.0218820, upper bound: 0.0218033
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 8.58
Output dim: 6, lower bound: -0.0218820, upper bound: 0.0218033
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 8.58
Output dim: 6, lower bound: -0.0214377, upper bound: 0.0215394
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 8.58
Output dim: 6, lower bound: -0.0214377, upper bound: 0.0215394

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0357068, 0.0356896
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 193

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0202784, upper bound: 0.0202766
time: 3.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0202784, upper bound: 0.0202766
time: 3.05 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0357033, 0.0356940
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213741, upper bound: 0.0213732
time: 3.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213733, upper bound: 0.0213738
time: 3.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0354410, 0.0354800
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 130

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0204114, upper bound: 0.0204129
time: 2.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0204114, upper bound: 0.0204129
time: 2.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0354995, 0.0354334
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 193

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0198377, upper bound: 0.0198377
time: 3.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0198377, upper bound: 0.0198377
time: 3.06 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0358195, 0.0358269
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
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0217478, upper bound: 0.0217061
time: 4.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0217072, upper bound: 0.0217429
time: 2.98 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0358198, 0.0358258
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
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 153

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 235

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0209608, upper bound: 0.0209546
time: 3.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0209608, upper bound: 0.0209546
time: 3.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0358071, 0.0357797
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213180, upper bound: 0.0212862
time: 3.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213180, upper bound: 0.0212862
time: 3.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0357998, 0.0357863
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
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 188

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0218140, upper bound: 0.0218360
time: 4.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0218112, upper bound: 0.0218447
time: 3.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0358187, 0.0358219
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
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0194048, upper bound: 0.0193972
time: 2.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0194048, upper bound: 0.0193972
time: 2.46 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0358151, 0.0358364
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
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0211224, upper bound: 0.0211113
time: 3.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0211224, upper bound: 0.0211113
time: 3.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0360474, 0.0360168
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
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0201599, upper bound: 0.0201532
time: 2.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0201599, upper bound: 0.0201532
time: 2.52 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0360358, 0.0360253
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
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 130

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0209619, upper bound: 0.0209587
time: 3.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0209619, upper bound: 0.0209587
time: 4.05 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0352694, 0.0351595
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
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0217716, upper bound: 0.0216251
time: 5.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0216023, upper bound: 0.0216853
time: 3.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0352698, 0.0351589
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
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213482, upper bound: 0.0212733
time: 3.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213482, upper bound: 0.0212733
time: 3.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0351204, 0.0352424
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
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0214196, upper bound: 0.0215213
time: 3.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0214202, upper bound: 0.0215212
time: 3.55 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0351526, 0.0352583
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
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0214053, upper bound: 0.0215059
time: 3.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0214052, upper bound: 0.0215066
time: 3.15 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 9.65 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 9.65
Output dim: 6, lower bound: -0.0202784, upper bound: 0.0202766
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 9.65
Output dim: 6, lower bound: -0.0202784, upper bound: 0.0202766
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.65
Output dim: 6, lower bound: -0.0213741, upper bound: 0.0213732
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.65
Output dim: 6, lower bound: -0.0213733, upper bound: 0.0213738
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 9.65
Output dim: 6, lower bound: -0.0204114, upper bound: 0.0204129
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 9.65
Output dim: 6, lower bound: -0.0204114, upper bound: 0.0204129
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 9.65
Output dim: 6, lower bound: -0.0198377, upper bound: 0.0198377
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 9.65
Output dim: 6, lower bound: -0.0198377, upper bound: 0.0198377
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.65
Output dim: 6, lower bound: -0.0217478, upper bound: 0.0217061
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.65
Output dim: 6, lower bound: -0.0217072, upper bound: 0.0217429
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.65
Output dim: 6, lower bound: -0.0209608, upper bound: 0.0209546
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.65
Output dim: 6, lower bound: -0.0209608, upper bound: 0.0209546
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.65
Output dim: 6, lower bound: -0.0213180, upper bound: 0.0212862
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.65
Output dim: 6, lower bound: -0.0213180, upper bound: 0.0212862
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.65
Output dim: 6, lower bound: -0.0218140, upper bound: 0.0218360
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.65
Output dim: 6, lower bound: -0.0218112, upper bound: 0.0218447
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 9.65
Output dim: 6, lower bound: -0.0194048, upper bound: 0.0193972
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 9.65
Output dim: 6, lower bound: -0.0194048, upper bound: 0.0193972
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.65
Output dim: 6, lower bound: -0.0211224, upper bound: 0.0211113
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.65
Output dim: 6, lower bound: -0.0211224, upper bound: 0.0211113
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 9.65
Output dim: 6, lower bound: -0.0201599, upper bound: 0.0201532
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 9.65
Output dim: 6, lower bound: -0.0201599, upper bound: 0.0201532
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.65
Output dim: 6, lower bound: -0.0209619, upper bound: 0.0209587
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.65
Output dim: 6, lower bound: -0.0209619, upper bound: 0.0209587
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.65
Output dim: 6, lower bound: -0.0217716, upper bound: 0.0216251
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.65
Output dim: 6, lower bound: -0.0216023, upper bound: 0.0216853
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.65
Output dim: 6, lower bound: -0.0213482, upper bound: 0.0212733
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.65
Output dim: 6, lower bound: -0.0213482, upper bound: 0.0212733
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.65
Output dim: 6, lower bound: -0.0214196, upper bound: 0.0215213
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.65
Output dim: 6, lower bound: -0.0214202, upper bound: 0.0215212
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.65
Output dim: 6, lower bound: -0.0214053, upper bound: 0.0215059
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.65
Output dim: 6, lower bound: -0.0214052, upper bound: 0.0215066

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0357022, 0.0357041
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
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0200442, upper bound: 0.0200425
time: 3.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0200442, upper bound: 0.0200425
time: 3.11 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0357207, 0.0356929
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
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0209538, upper bound: 0.0209534
time: 3.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0209538, upper bound: 0.0209534
time: 3.46 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0357820, 0.0357817
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 130

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0214566, upper bound: 0.0214202
time: 4.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0214566, upper bound: 0.0214202
time: 4.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0357743, 0.0357881
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
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 153

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0217071, upper bound: 0.0217211
time: 4.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0216931, upper bound: 0.0217429
time: 4.02 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0358076, 0.0358126
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0209608, upper bound: 0.0209361
time: 3.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0209433, upper bound: 0.0209546
time: 5.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0358198, 0.0358136
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
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0201817, upper bound: 0.0201775
time: 2.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0201817, upper bound: 0.0201775
time: 2.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0357741, 0.0357296
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
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0210737, upper bound: 0.0210448
time: 3.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0210737, upper bound: 0.0210448
time: 3.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0358071, 0.0357467
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
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 90

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0212663, upper bound: 0.0211863
time: 3.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0212212, upper bound: 0.0212351
time: 3.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0357829, 0.0357689
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
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0217152, upper bound: 0.0216738
time: 3.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0216559, upper bound: 0.0217381
time: 3.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0357825, 0.0357694
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
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 165

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0208459, upper bound: 0.0208749
time: 3.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0208459, upper bound: 0.0208749
time: 3.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0358151, 0.0358321
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
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 193

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0209109, upper bound: 0.0209001
time: 3.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0209050, upper bound: 0.0209042
time: 2.93 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0358108, 0.0358364
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
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 235

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 130

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0210907, upper bound: 0.0210792
time: 3.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0210907, upper bound: 0.0210793
time: 3.49 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0359845, 0.0359385
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
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 193

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0189954, upper bound: 0.0189906
time: 1.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0189954, upper bound: 0.0189906
time: 1.92 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0359490, 0.0359767
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
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 193

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 130

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0208510, upper bound: 0.0207925
time: 3.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0207936, upper bound: 0.0208486
time: 3.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0346580, 0.0344287
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
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0216957, upper bound: 0.0215399
time: 4.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0216916, upper bound: 0.0215498
time: 3.94 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0345386, 0.0345359
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
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0215470, upper bound: 0.0215305
time: 4.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0215265, upper bound: 0.0215737
time: 9.16 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0352349, 0.0351184
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
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0211252, upper bound: 0.0210321
time: 3.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0211106, upper bound: 0.0210525
time: 3.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0352293, 0.0351275
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
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 220

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0212462, upper bound: 0.0211146
time: 4.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0211799, upper bound: 0.0211717
time: 4.23 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0351022, 0.0352426
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
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0209698, upper bound: 0.0210601
time: 3.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0209698, upper bound: 0.0210601
time: 3.46 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0351214, 0.0352242
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
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 153

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0211519, upper bound: 0.0212398
time: 4.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0211340, upper bound: 0.0212517
time: 3.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0350333, 0.0351357
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
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 130

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0211388, upper bound: 0.0212268
time: 3.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0211206, upper bound: 0.0212386
time: 3.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0350279, 0.0351385
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
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 130

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0202294, upper bound: 0.0202876
time: 2.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0202294, upper bound: 0.0202876
time: 2.29 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 11.75 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 11.75
Output dim: 6, lower bound: -0.0200442, upper bound: 0.0200425
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 11.75
Output dim: 6, lower bound: -0.0200442, upper bound: 0.0200425
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 11.75
Output dim: 6, lower bound: -0.0209538, upper bound: 0.0209534
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 11.75
Output dim: 6, lower bound: -0.0209538, upper bound: 0.0209534
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 11.75
Output dim: 6, lower bound: -0.0214566, upper bound: 0.0214202
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 11.75
Output dim: 6, lower bound: -0.0214566, upper bound: 0.0214202
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 11.75
Output dim: 6, lower bound: -0.0217071, upper bound: 0.0217211
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 11.75
Output dim: 6, lower bound: -0.0216931, upper bound: 0.0217429
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 11.75
Output dim: 6, lower bound: -0.0209608, upper bound: 0.0209361
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 11.75
Output dim: 6, lower bound: -0.0209433, upper bound: 0.0209546
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 11.75
Output dim: 6, lower bound: -0.0201817, upper bound: 0.0201775
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 11.75
Output dim: 6, lower bound: -0.0201817, upper bound: 0.0201775
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 11.75
Output dim: 6, lower bound: -0.0210737, upper bound: 0.0210448
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 11.75
Output dim: 6, lower bound: -0.0210737, upper bound: 0.0210448
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 11.75
Output dim: 6, lower bound: -0.0212663, upper bound: 0.0211863
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 11.75
Output dim: 6, lower bound: -0.0212212, upper bound: 0.0212351
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 11.75
Output dim: 6, lower bound: -0.0217152, upper bound: 0.0216738
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 11.75
Output dim: 6, lower bound: -0.0216559, upper bound: 0.0217381
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 11.75
Output dim: 6, lower bound: -0.0208459, upper bound: 0.0208749
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 11.75
Output dim: 6, lower bound: -0.0208459, upper bound: 0.0208749
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 11.75
Output dim: 6, lower bound: -0.0209109, upper bound: 0.0209001
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 11.75
Output dim: 6, lower bound: -0.0209050, upper bound: 0.0209042
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 11.75
Output dim: 6, lower bound: -0.0210907, upper bound: 0.0210792
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 11.75
Output dim: 6, lower bound: -0.0210907, upper bound: 0.0210793
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 11.75
Output dim: 6, lower bound: -0.0189954, upper bound: 0.0189906
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 11.75
Output dim: 6, lower bound: -0.0189954, upper bound: 0.0189906
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 11.75
Output dim: 6, lower bound: -0.0208510, upper bound: 0.0207925
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 11.75
Output dim: 6, lower bound: -0.0207936, upper bound: 0.0208486
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 11.75
Output dim: 6, lower bound: -0.0216957, upper bound: 0.0215399
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 11.75
Output dim: 6, lower bound: -0.0216916, upper bound: 0.0215498
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 11.75
Output dim: 6, lower bound: -0.0215470, upper bound: 0.0215305
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 11.75
Output dim: 6, lower bound: -0.0215265, upper bound: 0.0215737
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 11.75
Output dim: 6, lower bound: -0.0211252, upper bound: 0.0210321
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 11.75
Output dim: 6, lower bound: -0.0211106, upper bound: 0.0210525
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 11.75
Output dim: 6, lower bound: -0.0212462, upper bound: 0.0211146
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 11.75
Output dim: 6, lower bound: -0.0211799, upper bound: 0.0211717
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 11.75
Output dim: 6, lower bound: -0.0209698, upper bound: 0.0210601
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 11.75
Output dim: 6, lower bound: -0.0209698, upper bound: 0.0210601
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 11.75
Output dim: 6, lower bound: -0.0211519, upper bound: 0.0212398
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 11.75
Output dim: 6, lower bound: -0.0211340, upper bound: 0.0212517
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 11.75
Output dim: 6, lower bound: -0.0211388, upper bound: 0.0212268
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 11.75
Output dim: 6, lower bound: -0.0211206, upper bound: 0.0212386
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 11.75
Output dim: 6, lower bound: -0.0202294, upper bound: 0.0202876
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 11.75
Output dim: 6, lower bound: -0.0202294, upper bound: 0.0202876

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0356496, 0.0355948
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0199049, upper bound: 0.0199052
time: 3.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0199049, upper bound: 0.0199052
time: 3.24 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0356301, 0.0356218
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0208195, upper bound: 0.0207644
time: 3.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0207639, upper bound: 0.0208191
time: 3.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0357105, 0.0356922
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0212610, upper bound: 0.0212032
time: 3.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0212374, upper bound: 0.0212229
time: 3.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0356871, 0.0357102
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213231, upper bound: 0.0212922
time: 3.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213231, upper bound: 0.0212922
time: 3.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0358913, 0.0358737
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0216994, upper bound: 0.0217138
time: 4.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0216994, upper bound: 0.0217138
time: 3.50 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0358402, 0.0359051
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0211059, upper bound: 0.0211440
time: 3.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0211022, upper bound: 0.0211502
time: 3.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0359240, 0.0358944
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
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0195900, upper bound: 0.0195766
time: 2.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0195900, upper bound: 0.0195766
time: 2.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0358729, 0.0359291
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
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0200753, upper bound: 0.0200866
time: 3.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0200753, upper bound: 0.0200866
time: 2.98 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0354495, 0.0354548
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0210434, upper bound: 0.0210147
time: 3.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0210434, upper bound: 0.0210140
time: 3.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0355241, 0.0354050
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
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0202681, upper bound: 0.0202432
time: 3.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0202681, upper bound: 0.0202433
time: 2.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0355625, 0.0354301
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0210237, upper bound: 0.0209359
time: 3.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0210237, upper bound: 0.0209359
time: 4.02 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0354906, 0.0354907
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 165

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0212054, upper bound: 0.0212191
time: 3.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0212044, upper bound: 0.0212194
time: 3.46 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0354863, 0.0352935
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0216728, upper bound: 0.0215340
time: 4.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0215664, upper bound: 0.0216311
time: 3.97 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0353075, 0.0354727
8: -0.0036455, 0.0094795, -0.0036455, 0.0094795, -0.0131250, 0.0131250
9: -0.0400144, -0.0022857, -0.0400144, -0.0022857, -0.0377288, 0.0377288

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0216286, upper bound: 0.0217041
time: 3.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0216271, upper bound: 0.0217085
time: 3.23 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0024000, 0.0091893, -0.0024000, 0.0091893, -0.0115893, 0.0115893
1: -0.0061213, 0.0058810, -0.0061213, 0.0058810, -0.0120023, 0.0120023
2: -0.0405399, 0.0183484, -0.0405399, 0.0183484, -0.0588883, 0.0588883
3: -0.0041514, 0.0208036, -0.0041514, 0.0208036, -0.0249549, 0.0249549
4: 0.0011945, 0.0223657, 0.0011945, 0.0223657, -0.0211712, 0.0211712
5: -0.0038585, 0.0280466, -0.0038585, 0.0280466, -0.0319050, 0.0319050
6: 0.9915334, 1.0163498, 0.9915334, 1.0163498, -0.0248164, 0.0248164
7: -0.0112205, 0.0271030, -0.0112205, 0.0271030, -0.0357825, 0.0357652
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
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0203761, upper bound: 0.0203960
time: 3.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0203761, upper bound: 0.0203960
time: 3.25 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 7.84 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 7.84
Output dim: 6, lower bound: -0.0199049, upper bound: 0.0199052
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 7.84
Output dim: 6, lower bound: -0.0199049, upper bound: 0.0199052
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.84
Output dim: 6, lower bound: -0.0208195, upper bound: 0.0207644
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.84
Output dim: 6, lower bound: -0.0207639, upper bound: 0.0208191
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.84
Output dim: 6, lower bound: -0.0212610, upper bound: 0.0212032
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.84
Output dim: 6, lower bound: -0.0212374, upper bound: 0.0212229
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.84
Output dim: 6, lower bound: -0.0213231, upper bound: 0.0212922
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.84
Output dim: 6, lower bound: -0.0213231, upper bound: 0.0212922
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.84
Output dim: 6, lower bound: -0.0216994, upper bound: 0.0217138
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.84
Output dim: 6, lower bound: -0.0216994, upper bound: 0.0217138
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.84
Output dim: 6, lower bound: -0.0211059, upper bound: 0.0211440
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.84
Output dim: 6, lower bound: -0.0211022, upper bound: 0.0211502
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 7.84
Output dim: 6, lower bound: -0.0195900, upper bound: 0.0195766
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 7.84
Output dim: 6, lower bound: -0.0195900, upper bound: 0.0195766
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 7.84
Output dim: 6, lower bound: -0.0200753, upper bound: 0.0200866
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 7.84
Output dim: 6, lower bound: -0.0200753, upper bound: 0.0200866
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.84
Output dim: 6, lower bound: -0.0210434, upper bound: 0.0210147
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.84
Output dim: 6, lower bound: -0.0210434, upper bound: 0.0210140
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 7.84
Output dim: 6, lower bound: -0.0202681, upper bound: 0.0202432
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 7.84
Output dim: 6, lower bound: -0.0202681, upper bound: 0.0202433
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.84
Output dim: 6, lower bound: -0.0210237, upper bound: 0.0209359
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.84
Output dim: 6, lower bound: -0.0210237, upper bound: 0.0209359
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.84
Output dim: 6, lower bound: -0.0212054, upper bound: 0.0212191
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.84
Output dim: 6, lower bound: -0.0212044, upper bound: 0.0212194
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.84
Output dim: 6, lower bound: -0.0216728, upper bound: 0.0215340
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.84
Output dim: 6, lower bound: -0.0215664, upper bound: 0.0216311
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.84
Output dim: 6, lower bound: -0.0216286, upper bound: 0.0217041
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.84
Output dim: 6, lower bound: -0.0216271, upper bound: 0.0217085
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 7.84
Output dim: 6, lower bound: -0.0203761, upper bound: 0.0203960
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 7.84
Output dim: 6, lower bound: -0.0203761, upper bound: 0.0203960
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.84
Output dim: 6, lower bound: -0.0208459, upper bound: 0.0208749
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.84
Output dim: 6, lower bound: -0.0209109, upper bound: 0.0209001
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.84
Output dim: 6, lower bound: -0.0209050, upper bound: 0.0209042
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.84
Output dim: 6, lower bound: -0.0210907, upper bound: 0.0210792
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.84
Output dim: 6, lower bound: -0.0210907, upper bound: 0.0210793
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.84
Output dim: 6, lower bound: -0.0208510, upper bound: 0.0207925
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.84
Output dim: 6, lower bound: -0.0207936, upper bound: 0.0208486
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.84
Output dim: 6, lower bound: -0.0216957, upper bound: 0.0215399
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.84
Output dim: 6, lower bound: -0.0216916, upper bound: 0.0215498
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.84
Output dim: 6, lower bound: -0.0215470, upper bound: 0.0215305
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.84
Output dim: 6, lower bound: -0.0215265, upper bound: 0.0215737
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.84
Output dim: 6, lower bound: -0.0211252, upper bound: 0.0210321
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.84
Output dim: 6, lower bound: -0.0211106, upper bound: 0.0210525
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.84
Output dim: 6, lower bound: -0.0212462, upper bound: 0.0211146
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.84
Output dim: 6, lower bound: -0.0211799, upper bound: 0.0211717
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.84
Output dim: 6, lower bound: -0.0209698, upper bound: 0.0210601
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.84
Output dim: 6, lower bound: -0.0209698, upper bound: 0.0210601
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.84
Output dim: 6, lower bound: -0.0211519, upper bound: 0.0212398
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.84
Output dim: 6, lower bound: -0.0211340, upper bound: 0.0212517
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.84
Output dim: 6, lower bound: -0.0211388, upper bound: 0.0212268
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.84
Output dim: 6, lower bound: -0.0211206, upper bound: 0.0212386

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 6.52 + 597.97 = 604.49 seconds
