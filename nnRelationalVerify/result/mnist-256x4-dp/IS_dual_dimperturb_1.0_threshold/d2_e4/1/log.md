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
execution time: IAR + RelationalAnalysis = 1.33 + 5.20 = 6.53 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.0221068, upper bound: 0.0221068

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0216167, upper bound: 0.0215854
time: 3.36 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0215865, upper bound: 0.0215864
time: 3.44 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 6.94 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 6.94
Output dim: 6, lower bound: -0.0216167, upper bound: 0.0215854
IS_A2, status: Status.UNKNOWN, split count: 1, time: 6.94
Output dim: 6, lower bound: -0.0215865, upper bound: 0.0215864

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0018476, 0.0091385, -0.0023548, 0.0091852, -0.0110328, 0.0114933
1: -0.0055951, 0.0057825, -0.0060783, 0.0058729, -0.0114680, 0.0118607
2: -0.0392230, 0.0180601, -0.0404322, 0.0183253, -0.0575483, 0.0584923
3: -0.0041256, 0.0196608, -0.0041493, 0.0207101, -0.0248358, 0.0238101
4: 0.0013195, 0.0220213, 0.0012046, 0.0223376, -0.0210181, 0.0208168
5: -0.0038071, 0.0264373, -0.0038543, 0.0279150, -0.0317221, 0.0302915
6: 0.9915676, 1.0152655, 0.9915361, 1.0162612, -0.0246937, 0.0237294
7: -0.0109944, 0.0264796, -0.0112024, 0.0270520, -0.0359156, 0.0356230
8: -0.0031577, 0.0092842, -0.0036057, 0.0094635, -0.0126212, 0.0128899
9: -0.0389081, -0.0024271, -0.0399240, -0.0022970, -0.0366111, 0.0374969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=38, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=73, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 198

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0207512, upper bound: 0.0211841
time: 3.23 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0214938, upper bound: 0.0214660
time: 4.09 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0016767, 0.0091227, -0.0020508, 0.0091572, -0.0108338, 0.0111735
1: -0.0054323, 0.0057520, -0.0057886, 0.0058187, -0.0112510, 0.0115406
2: -0.0388155, 0.0204807, -0.0397074, 0.0180726, -0.0568881, 0.0601881
3: -0.0043418, 0.0193071, -0.0041267, 0.0200811, -0.0244229, 0.0234339
4: 0.0002705, 0.0219148, 0.0013141, 0.0221480, -0.0218775, 0.0206007
5: -0.0037912, 0.0259393, -0.0038260, 0.0270292, -0.0308203, 0.0297652
6: 0.9912804, 1.0149300, 0.9915661, 1.0156643, -0.0243840, 0.0233639
7: -0.0128931, 0.0262866, -0.0110042, 0.0267088, -0.0373830, 0.0353898
8: -0.0030510, 0.0092238, -0.0033372, 0.0093560, -0.0124070, 0.0125609
9: -0.0385658, -0.0012398, -0.0393150, -0.0024209, -0.0361448, 0.0380753

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=38, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=74, inp2_unstable=73, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 198

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0207161, upper bound: 0.0212035
time: 3.42 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0214600, upper bound: 0.0214600
time: 10.55 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 15.31 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 15.31
Output dim: 6, lower bound: -0.0207512, upper bound: 0.0211841
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 15.31
Output dim: 6, lower bound: -0.0214938, upper bound: 0.0214660
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 15.31
Output dim: 6, lower bound: -0.0207161, upper bound: 0.0212035
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 15.31
Output dim: 6, lower bound: -0.0214600, upper bound: 0.0214600

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0015676, 0.0091127, -0.0017266, 0.0091273, -0.0106949, 0.0108392
1: -0.0053283, 0.0057325, -0.0054798, 0.0057609, -0.0110892, 0.0112123
2: -0.0385552, 0.0178629, -0.0389343, 0.0175743, -0.0561295, 0.0567972
3: -0.0041080, 0.0190813, -0.0040822, 0.0194103, -0.0235183, 0.0231636
4: 0.0014049, 0.0218467, 0.0015300, 0.0219458, -0.0205409, 0.0203167
5: -0.0037810, 0.0256213, -0.0037958, 0.0260845, -0.0298655, 0.0294171
6: 0.9915910, 1.0147157, 0.9916252, 1.0150278, -0.0234368, 0.0230904
7: -0.0108397, 0.0261635, -0.0106133, 0.0263429, -0.0350668, 0.0347032
8: -0.0029103, 0.0091852, -0.0030508, 0.0092414, -0.0121517, 0.0122359
9: -0.0383472, -0.0025238, -0.0386656, -0.0026654, -0.0356818, 0.0361418

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=74, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 188

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0198481, upper bound: 0.0202672
time: 3.33 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0197665, upper bound: 0.0202008
time: 2.80 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0017655, 0.0091309, -0.0021350, 0.0091649, -0.0109304, 0.0112659
1: -0.0055169, 0.0057678, -0.0058688, 0.0058337, -0.0113506, 0.0116367
2: -0.0390272, 0.0179537, -0.0399081, 0.0180279, -0.0570551, 0.0578617
3: -0.0041161, 0.0194909, -0.0041227, 0.0202553, -0.0243714, 0.0236136
4: 0.0013656, 0.0219701, 0.0013334, 0.0222005, -0.0208349, 0.0206367
5: -0.0037994, 0.0261980, -0.0038338, 0.0272745, -0.0310739, 0.0300318
6: 0.9915802, 1.0151043, 0.9915714, 1.0158296, -0.0242493, 0.0235329
7: -0.0109109, 0.0263869, -0.0109691, 0.0268039, -0.0350842, 0.0352936
8: -0.0030852, 0.0092552, -0.0034115, 0.0093858, -0.0124710, 0.0126666
9: -0.0387436, -0.0024793, -0.0394837, -0.0024429, -0.0363008, 0.0370044

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=73, inp2_unstable=74, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0208043, upper bound: 0.0207677
time: 3.22 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0205288, upper bound: 0.0205649
time: 2.91 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0013926, 0.0090966, -0.0014263, 0.0090997, -0.0104923, 0.0105228
1: -0.0051616, 0.0057013, -0.0051937, 0.0057073, -0.0108689, 0.0108950
2: -0.0381381, 0.0202892, -0.0382184, 0.0173258, -0.0554638, 0.0585076
3: -0.0043247, 0.0187193, -0.0040600, 0.0187891, -0.0231138, 0.0227794
4: 0.0003535, 0.0217376, 0.0016377, 0.0217586, -0.0214051, 0.0200999
5: -0.0037647, 0.0251115, -0.0037678, 0.0252098, -0.0289745, 0.0288794
6: 0.9913031, 1.0143722, 0.9916547, 1.0144385, -0.0231354, 0.0227175
7: -0.0127429, 0.0259660, -0.0104183, 0.0260040, -0.0365353, 0.0344727
8: -0.0030039, 0.0091233, -0.0027855, 0.0091352, -0.0121391, 0.0119088
9: -0.0379967, -0.0013337, -0.0380642, -0.0027873, -0.0352095, 0.0367305

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 193

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0197898, upper bound: 0.0202414
time: 3.25 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0196761, upper bound: 0.0201262
time: 2.58 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0015937, 0.0091151, -0.0018263, 0.0091365, -0.0107302, 0.0109414
1: -0.0053532, 0.0057372, -0.0055748, 0.0057786, -0.0111318, 0.0113119
2: -0.0386175, 0.0203774, -0.0391721, 0.0177765, -0.0563940, 0.0595495
3: -0.0043326, 0.0191354, -0.0041003, 0.0196167, -0.0239493, 0.0232357
4: 0.0003153, 0.0218630, 0.0014424, 0.0220080, -0.0216927, 0.0204206
5: -0.0037834, 0.0256974, -0.0038051, 0.0263751, -0.0301586, 0.0295024
6: 0.9912927, 1.0147669, 0.9916012, 1.0152236, -0.0239310, 0.0231657
7: -0.0128121, 0.0261929, -0.0107719, 0.0264555, -0.0366018, 0.0350576
8: -0.0030256, 0.0091944, -0.0031389, 0.0092767, -0.0123023, 0.0123333
9: -0.0383995, -0.0012905, -0.0388654, -0.0025662, -0.0358333, 0.0375750

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=73, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_B1

### Relational analysis result of IS_A2_B2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0206350, upper bound: 0.0205590
time: 3.35 seconds

## Relational analysis of IS_A2_B2_B2

### Relational analysis result of IS_A2_B2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0204351, upper bound: 0.0204351
time: 2.41 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 7.12 seconds
IS_A1_B1_A1, status: Status.VERIFIED, split count: 3, time: 7.12
Output dim: 6, lower bound: -0.0198481, upper bound: 0.0202672
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 7.12
Output dim: 6, lower bound: -0.0197665, upper bound: 0.0202008
IS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 7.12
Output dim: 6, lower bound: -0.0208043, upper bound: 0.0207677
IS_A1_B2_B2, status: Status.VERIFIED, split count: 3, time: 7.12
Output dim: 6, lower bound: -0.0205288, upper bound: 0.0205649
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 7.12
Output dim: 6, lower bound: -0.0197898, upper bound: 0.0202414
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 7.12
Output dim: 6, lower bound: -0.0196761, upper bound: 0.0201262
IS_A2_B2_B1, status: Status.VERIFIED, split count: 3, time: 7.12
Output dim: 6, lower bound: -0.0206350, upper bound: 0.0205590
IS_A2_B2_B2, status: Status.VERIFIED, split count: 3, time: 7.12
Output dim: 6, lower bound: -0.0204351, upper bound: 0.0204351

## BFS IS instance: IS_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0017655, 0.0091309, -0.0021139, 0.0091630, -0.0109285, 0.0112448
1: -0.0055169, 0.0057678, -0.0058488, 0.0058300, -0.0113468, 0.0116166
2: -0.0390272, 0.0179537, -0.0398579, 0.0174263, -0.0564534, 0.0578115
3: -0.0041161, 0.0194909, -0.0040690, 0.0202117, -0.0243278, 0.0235599
4: 0.0013656, 0.0219701, 0.0015941, 0.0221874, -0.0208218, 0.0203760
5: -0.0037994, 0.0261980, -0.0038318, 0.0272131, -0.0310126, 0.0300298
6: 0.9915802, 1.0151043, 0.9916428, 1.0157883, -0.0242081, 0.0234615
7: -0.0109109, 0.0263869, -0.0104971, 0.0267801, -0.0350589, 0.0347158
8: -0.0030852, 0.0092552, -0.0033929, 0.0093784, -0.0124636, 0.0126480
9: -0.0387436, -0.0024793, -0.0394415, -0.0027380, -0.0360057, 0.0369622

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=36, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=73, inp2_unstable=74, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B2_B1_A1

### Relational analysis result of IS_A1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0202019, upper bound: 0.0201318
time: 3.19 seconds

## Relational analysis of IS_A1_B2_B1_A2

### Relational analysis result of IS_A1_B2_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0200207, upper bound: 0.0199134
time: 2.60 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 7.13 seconds
IS_A1_B2_B1_A1, status: Status.VERIFIED, split count: 4, time: 7.13
Output dim: 6, lower bound: -0.0202019, upper bound: 0.0201318
IS_A1_B2_B1_A2, status: Status.VERIFIED, split count: 4, time: 7.13
Output dim: 6, lower bound: -0.0200207, upper bound: 0.0199134

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 6.53 + 67.34 = 73.87 seconds
