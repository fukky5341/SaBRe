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
execution time: IAR + RelationalAnalysis = 1.35 + 5.16 = 6.52 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.0221068, upper bound: 0.0221068

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 188

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0216167, upper bound: 0.0215854
time: 3.35 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0215865, upper bound: 0.0215864
time: 3.43 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 6.90 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 6.90
Output dim: 6, lower bound: -0.0216167, upper bound: 0.0215854
IS_A2, status: Status.UNKNOWN, split count: 1, time: 6.90
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

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 188

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0215683, upper bound: 0.0215683
time: 3.68 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0215683, upper bound: 0.0215683
time: 4.35 seconds

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
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 188

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0207819, upper bound: 0.0207080
time: 3.05 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0205898, upper bound: 0.0205899
time: 2.43 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 6.80 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 6.80
Output dim: 6, lower bound: -0.0215683, upper bound: 0.0215683
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 6.80
Output dim: 6, lower bound: -0.0215683, upper bound: 0.0215683
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 6.80
Output dim: 6, lower bound: -0.0207819, upper bound: 0.0207080
IS_A2_B2, status: Status.VERIFIED, split count: 2, time: 6.80
Output dim: 6, lower bound: -0.0205898, upper bound: 0.0205899

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0018476, 0.0091385, -0.0018476, 0.0091385, -0.0109861, 0.0109861
1: -0.0055951, 0.0057825, -0.0055951, 0.0057825, -0.0113776, 0.0113776
2: -0.0392230, 0.0180601, -0.0392230, 0.0180601, -0.0572831, 0.0572831
3: -0.0041256, 0.0196608, -0.0041256, 0.0196608, -0.0237864, 0.0237864
4: 0.0013195, 0.0220213, 0.0013195, 0.0220213, -0.0207019, 0.0207019
5: -0.0038071, 0.0264373, -0.0038071, 0.0264373, -0.0302443, 0.0302443
6: 0.9915676, 1.0152655, 0.9915676, 1.0152655, -0.0236979, 0.0236979
7: -0.0109944, 0.0264796, -0.0109944, 0.0264796, -0.0354122, 0.0354122
8: -0.0031577, 0.0092842, -0.0031577, 0.0092842, -0.0124419, 0.0124419
9: -0.0389081, -0.0024271, -0.0389081, -0.0024271, -0.0364810, 0.0364810

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=73, inp2_unstable=73, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 188

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0207652, upper bound: 0.0208129
time: 2.55 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0206805, upper bound: 0.0207116
time: 2.89 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0018476, 0.0091385, -0.0016767, 0.0091227, -0.0109704, 0.0108151
1: -0.0055951, 0.0057825, -0.0054323, 0.0057520, -0.0113471, 0.0112147
2: -0.0392230, 0.0180601, -0.0388155, 0.0204807, -0.0597037, 0.0568756
3: -0.0041256, 0.0196608, -0.0043418, 0.0193071, -0.0234328, 0.0240026
4: 0.0013195, 0.0220213, 0.0002705, 0.0219148, -0.0205953, 0.0217508
5: -0.0038071, 0.0264373, -0.0037912, 0.0259393, -0.0297463, 0.0302284
6: 0.9915676, 1.0152655, 0.9912804, 1.0149300, -0.0233625, 0.0239851
7: -0.0109944, 0.0264796, -0.0128931, 0.0262866, -0.0351851, 0.0372442
8: -0.0031577, 0.0092842, -0.0030510, 0.0092238, -0.0123815, 0.0123352
9: -0.0389081, -0.0024271, -0.0385658, -0.0012398, -0.0376683, 0.0361387

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=73, inp2_unstable=74, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 188

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0207652, upper bound: 0.0208128
time: 3.42 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0206805, upper bound: 0.0207116
time: 2.70 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0016767, 0.0091227, -0.0020300, 0.0091553, -0.0108319, 0.0111527
1: -0.0054323, 0.0057520, -0.0057689, 0.0058150, -0.0112473, 0.0115208
2: -0.0388155, 0.0204807, -0.0396578, 0.0174708, -0.0562863, 0.0601386
3: -0.0043418, 0.0193071, -0.0040730, 0.0200382, -0.0243800, 0.0233801
4: 0.0002705, 0.0219148, 0.0015748, 0.0221351, -0.0218645, 0.0203399
5: -0.0037912, 0.0259393, -0.0038240, 0.0269687, -0.0307599, 0.0297633
6: 0.9912804, 1.0149300, 0.9916375, 1.0156236, -0.0243432, 0.0232925
7: -0.0128931, 0.0262866, -0.0105321, 0.0266854, -0.0373579, 0.0348101
8: -0.0030510, 0.0092238, -0.0033188, 0.0093487, -0.0123997, 0.0125425
9: -0.0385658, -0.0012398, -0.0392735, -0.0027161, -0.0358497, 0.0380337

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=74, inp2_unstable=73, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 188

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0205899, upper bound: 0.0205898
time: 2.91 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0205899, upper bound: 0.0205898
time: 2.64 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 6.90 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 6.90
Output dim: 6, lower bound: -0.0207652, upper bound: 0.0208129
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 6.90
Output dim: 6, lower bound: -0.0206805, upper bound: 0.0207116
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 6.90
Output dim: 6, lower bound: -0.0207652, upper bound: 0.0208128
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 6.90
Output dim: 6, lower bound: -0.0206805, upper bound: 0.0207116
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 6.90
Output dim: 6, lower bound: -0.0205899, upper bound: 0.0205898
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 6.90
Output dim: 6, lower bound: -0.0205899, upper bound: 0.0205898

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0018268, 0.0091365, -0.0018476, 0.0091385, -0.0109653, 0.0109842
1: -0.0055753, 0.0057787, -0.0055951, 0.0057825, -0.0113578, 0.0113739
2: -0.0391734, 0.0174575, -0.0392230, 0.0180601, -0.0572335, 0.0566804
3: -0.0040718, 0.0196178, -0.0041256, 0.0196608, -0.0237326, 0.0237434
4: 0.0015806, 0.0220084, 0.0013195, 0.0220213, -0.0204407, 0.0206889
5: -0.0038051, 0.0263767, -0.0038071, 0.0264373, -0.0302424, 0.0301838
6: 0.9916391, 1.0152247, 0.9915676, 1.0152655, -0.0236264, 0.0236571
7: -0.0105216, 0.0264561, -0.0109944, 0.0264796, -0.0348322, 0.0353872
8: -0.0031393, 0.0092769, -0.0031577, 0.0092842, -0.0124235, 0.0124346
9: -0.0388665, -0.0027227, -0.0389081, -0.0024271, -0.0364394, 0.0361855

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=37, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=73, inp2_unstable=73, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 188

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0212021, upper bound: 0.0212021
time: 3.58 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0212021, upper bound: 0.0212021
time: 3.58 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0018268, 0.0091365, -0.0016767, 0.0091227, -0.0109496, 0.0108132
1: -0.0055753, 0.0057787, -0.0054323, 0.0057520, -0.0113273, 0.0112110
2: -0.0391734, 0.0174575, -0.0388155, 0.0204807, -0.0596542, 0.0562729
3: -0.0040718, 0.0196178, -0.0043418, 0.0193071, -0.0233789, 0.0239596
4: 0.0015806, 0.0220084, 0.0002705, 0.0219148, -0.0203341, 0.0217379
5: -0.0038051, 0.0263767, -0.0037912, 0.0259393, -0.0297444, 0.0301679
6: 0.9916391, 1.0152247, 0.9912804, 1.0149300, -0.0232909, 0.0239443
7: -0.0105216, 0.0264561, -0.0128931, 0.0262866, -0.0346052, 0.0372192
8: -0.0031393, 0.0092769, -0.0030510, 0.0092238, -0.0123631, 0.0123279
9: -0.0388665, -0.0027227, -0.0385658, -0.0012398, -0.0376267, 0.0358431

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=37, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=73, inp2_unstable=74, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 188

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0206805, upper bound: 0.0207116
time: 2.82 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0206805, upper bound: 0.0207116
time: 2.88 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 7.06 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 7.06
Output dim: 6, lower bound: -0.0212021, upper bound: 0.0212021
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 7.06
Output dim: 6, lower bound: -0.0212021, upper bound: 0.0212021
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 7.06
Output dim: 6, lower bound: -0.0206805, upper bound: 0.0207116
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 7.06
Output dim: 6, lower bound: -0.0206805, upper bound: 0.0207116

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0018268, 0.0091365, -0.0018268, 0.0091365, -0.0109634, 0.0109634
1: -0.0055753, 0.0057787, -0.0055753, 0.0057787, -0.0113541, 0.0113541
2: -0.0391734, 0.0174575, -0.0391734, 0.0174575, -0.0566309, 0.0566309
3: -0.0040718, 0.0196178, -0.0040718, 0.0196178, -0.0236896, 0.0236896
4: 0.0015806, 0.0220084, 0.0015806, 0.0220084, -0.0204277, 0.0204277
5: -0.0038051, 0.0263767, -0.0038051, 0.0263767, -0.0301818, 0.0301818
6: 0.9916391, 1.0152247, 0.9916391, 1.0152247, -0.0235856, 0.0235856
7: -0.0105216, 0.0264561, -0.0105216, 0.0264561, -0.0348072, 0.0348072
8: -0.0031393, 0.0092769, -0.0031393, 0.0092769, -0.0124162, 0.0124162
9: -0.0388665, -0.0027227, -0.0388665, -0.0027227, -0.0361438, 0.0361438

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=73, inp2_unstable=73, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 188

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0208834, upper bound: 0.0203777
time: 4.05 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0211580, upper bound: 0.0211645
time: 3.20 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0018268, 0.0091365, -0.0036049, 0.0093003, -0.0111271, 0.0127415
1: -0.0055753, 0.0057787, -0.0072692, 0.0060959, -0.0116712, 0.0130479
2: -0.0391734, 0.0174575, -0.0434126, 0.0177073, -0.0568807, 0.0608701
3: -0.0040718, 0.0196178, -0.0040941, 0.0232965, -0.0273682, 0.0237119
4: 0.0015806, 0.0220084, 0.0014724, 0.0231170, -0.0215363, 0.0205360
5: -0.0038051, 0.0263767, -0.0039706, 0.0315570, -0.0353622, 0.0303473
6: 0.9916391, 1.0152247, 0.9916095, 1.0187151, -0.0270761, 0.0236152
7: -0.0105216, 0.0264561, -0.0107176, 0.0284628, -0.0372068, 0.0356383
8: -0.0031393, 0.0092769, -0.0047098, 0.0099056, -0.0130449, 0.0139867
9: -0.0388665, -0.0027227, -0.0424278, -0.0026001, -0.0362664, 0.0397051

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=73, inp2_unstable=79, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 188

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0208834, upper bound: 0.0203777
time: 2.57 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0211580, upper bound: 0.0211645
time: 2.77 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 6.69 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.69
Output dim: 6, lower bound: -0.0208834, upper bound: 0.0203777
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.69
Output dim: 6, lower bound: -0.0211580, upper bound: 0.0211645
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.69
Output dim: 6, lower bound: -0.0208834, upper bound: 0.0203777
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.69
Output dim: 6, lower bound: -0.0211580, upper bound: 0.0211645

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0011996, 0.0090788, -0.0015469, 0.0091108, -0.0103103, 0.0106257
1: -0.0049777, 0.0056669, -0.0053086, 0.0057288, -0.0107065, 0.0109754
2: -0.0376779, 0.0167130, -0.0385059, 0.0172606, -0.0549385, 0.0552190
3: -0.0040053, 0.0183200, -0.0040542, 0.0190386, -0.0230438, 0.0223742
4: 0.0019032, 0.0216173, 0.0016659, 0.0218338, -0.0199306, 0.0199513
5: -0.0037467, 0.0245491, -0.0037791, 0.0255611, -0.0293078, 0.0283282
6: 0.9917274, 1.0139933, 0.9916624, 1.0146751, -0.0229477, 0.0223309
7: -0.0099377, 0.0257481, -0.0103672, 0.0261401, -0.0338961, 0.0339474
8: -0.0025853, 0.0090551, -0.0028920, 0.0091779, -0.0117632, 0.0119471
9: -0.0376101, -0.0030878, -0.0383057, -0.0028192, -0.0347909, 0.0352180

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=36, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=71, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 188

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0210071, upper bound: 0.0205071
time: 3.29 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0210067, upper bound: 0.0204928
time: 3.67 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0016049, 0.0091161, -0.0017447, 0.0091290, -0.0107339, 0.0108608
1: -0.0053639, 0.0057392, -0.0054971, 0.0057641, -0.0111280, 0.0112362
2: -0.0386443, 0.0171619, -0.0389776, 0.0173510, -0.0559953, 0.0561395
3: -0.0040454, 0.0191586, -0.0040623, 0.0194479, -0.0234932, 0.0232209
4: 0.0017087, 0.0218700, 0.0016268, 0.0219572, -0.0202485, 0.0202432
5: -0.0037845, 0.0257302, -0.0037975, 0.0261374, -0.0299219, 0.0295276
6: 0.9916741, 1.0147891, 0.9916517, 1.0150634, -0.0233893, 0.0231375
7: -0.0102898, 0.0262056, -0.0104381, 0.0263634, -0.0344792, 0.0339747
8: -0.0029433, 0.0091984, -0.0030668, 0.0092478, -0.0121911, 0.0122652
9: -0.0384220, -0.0028676, -0.0387020, -0.0027749, -0.0356471, 0.0358344

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=36, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=71, inp2_unstable=73, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 188

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0214522, upper bound: 0.0214676
time: 3.61 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0214441, upper bound: 0.0214441
time: 4.56 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0011996, 0.0090788, -0.0033287, 0.0092749, -0.0104744, 0.0124075
1: -0.0049777, 0.0056669, -0.0070060, 0.0060466, -0.0110243, 0.0126728
2: -0.0376779, 0.0167130, -0.0427540, 0.0175083, -0.0551862, 0.0594671
3: -0.0040053, 0.0183200, -0.0040763, 0.0227249, -0.0267302, 0.0223963
4: 0.0019032, 0.0216173, 0.0015586, 0.0229447, -0.0210415, 0.0200587
5: -0.0037467, 0.0245491, -0.0039449, 0.0307522, -0.0344989, 0.0284940
6: 0.9917274, 1.0139933, 0.9916330, 1.0181727, -0.0264453, 0.0223603
7: -0.0099377, 0.0257481, -0.0105615, 0.0281511, -0.0363150, 0.0347812
8: -0.0025853, 0.0090551, -0.0044658, 0.0098079, -0.0123932, 0.0135208
9: -0.0376101, -0.0030878, -0.0418744, -0.0026977, -0.0349124, 0.0387867

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=36, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=71, inp2_unstable=79, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 193

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 92

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0207158, upper bound: 0.0201590
time: 3.30 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0207157, upper bound: 0.0201568
time: 3.94 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0016049, 0.0091161, -0.0035216, 0.0092926, -0.0108975, 0.0126377
1: -0.0053639, 0.0057392, -0.0071898, 0.0060811, -0.0114449, 0.0129290
2: -0.0386443, 0.0171619, -0.0432140, 0.0176014, -0.0562457, 0.0603759
3: -0.0040454, 0.0191586, -0.0040846, 0.0231241, -0.0271695, 0.0232433
4: 0.0017087, 0.0218700, 0.0015183, 0.0230650, -0.0213563, 0.0203517
5: -0.0037845, 0.0257302, -0.0039629, 0.0313144, -0.0350988, 0.0296930
6: 0.9916741, 1.0147891, 0.9916220, 1.0185516, -0.0268775, 0.0231671
7: -0.0102898, 0.0262056, -0.0106345, 0.0283688, -0.0368732, 0.0348471
8: -0.0029433, 0.0091984, -0.0046363, 0.0098761, -0.0128194, 0.0138346
9: -0.0384220, -0.0028676, -0.0422609, -0.0026521, -0.0357699, 0.0393933

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=36, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=71, inp2_unstable=79, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 193

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 92

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0211040, upper bound: 0.0211142
time: 3.89 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0211031, upper bound: 0.0211131
time: 2.94 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 8.19 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.19
Output dim: 6, lower bound: -0.0210071, upper bound: 0.0205071
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.19
Output dim: 6, lower bound: -0.0210067, upper bound: 0.0204928
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.19
Output dim: 6, lower bound: -0.0214522, upper bound: 0.0214676
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.19
Output dim: 6, lower bound: -0.0214441, upper bound: 0.0214441
IS_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 8.19
Output dim: 6, lower bound: -0.0207158, upper bound: 0.0201590
IS_A1_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 8.19
Output dim: 6, lower bound: -0.0207157, upper bound: 0.0201568
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.19
Output dim: 6, lower bound: -0.0211040, upper bound: 0.0211142
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.19
Output dim: 6, lower bound: -0.0211031, upper bound: 0.0211131

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0011876, 0.0090777, -0.0014886, 0.0091054, -0.0102930, 0.0105663
1: -0.0049663, 0.0056647, -0.0052531, 0.0057184, -0.0106847, 0.0109178
2: -0.0376493, 0.0165829, -0.0383670, 0.0166557, -0.0543050, 0.0549499
3: -0.0039937, 0.0182952, -0.0040002, 0.0189180, -0.0229117, 0.0222954
4: 0.0019596, 0.0216098, 0.0019281, 0.0217975, -0.0198379, 0.0196817
5: -0.0037456, 0.0245143, -0.0037736, 0.0253913, -0.0291369, 0.0282879
6: 0.9917429, 1.0139699, 0.9917343, 1.0145607, -0.0228178, 0.0222356
7: -0.0098356, 0.0257346, -0.0098927, 0.0260744, -0.0336235, 0.0333252
8: -0.0025747, 0.0090508, -0.0028406, 0.0091573, -0.0117319, 0.0118914
9: -0.0375861, -0.0031516, -0.0381890, -0.0031159, -0.0344702, 0.0350374

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=71, inp2_unstable=71, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 188

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 92

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0208814, upper bound: 0.0203548
time: 3.35 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0208272, upper bound: 0.0202831
time: 3.64 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0011533, 0.0090745, -0.0021277, 0.0091642, -0.0103176, 0.0112022
1: -0.0049337, 0.0056586, -0.0058618, 0.0058324, -0.0107661, 0.0115204
2: -0.0375676, 0.0161124, -0.0398905, 0.0177737, -0.0553413, 0.0560030
3: -0.0039517, 0.0182243, -0.0041000, 0.0202401, -0.0241917, 0.0223244
4: 0.0021635, 0.0215884, 0.0014436, 0.0221959, -0.0200324, 0.0201449
5: -0.0037424, 0.0244144, -0.0038331, 0.0272530, -0.0309955, 0.0282476
6: 0.9917986, 1.0139025, 0.9916016, 1.0158151, -0.0240166, 0.0223010
7: -0.0094666, 0.0256959, -0.0107697, 0.0267956, -0.0340573, 0.0353108
8: -0.0025444, 0.0090387, -0.0034050, 0.0093832, -0.0119277, 0.0124437
9: -0.0375175, -0.0033824, -0.0394689, -0.0025675, -0.0349500, 0.0360865

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=71, inp2_unstable=78, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 188

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 92

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0208814, upper bound: 0.0203423
time: 3.67 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0208266, upper bound: 0.0202674
time: 3.21 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0015927, 0.0091150, -0.0016858, 0.0091236, -0.0107162, 0.0108007
1: -0.0053522, 0.0057370, -0.0054409, 0.0057536, -0.0111058, 0.0111779
2: -0.0386151, 0.0170341, -0.0388370, 0.0167454, -0.0553605, 0.0558712
3: -0.0040340, 0.0191333, -0.0040082, 0.0193259, -0.0233599, 0.0231415
4: 0.0017641, 0.0218624, 0.0018892, 0.0219204, -0.0201563, 0.0199732
5: -0.0037833, 0.0256945, -0.0037920, 0.0259656, -0.0297490, 0.0294865
6: 0.9916893, 1.0147650, 0.9917236, 1.0149478, -0.0232585, 0.0230414
7: -0.0101895, 0.0261918, -0.0099631, 0.0262969, -0.0342078, 0.0333466
8: -0.0029325, 0.0091941, -0.0030147, 0.0092270, -0.0121595, 0.0122087
9: -0.0383975, -0.0029303, -0.0385839, -0.0030719, -0.0353256, 0.0356536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=71, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 188

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 92

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0214082, upper bound: 0.0214180
time: 3.20 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0214038, upper bound: 0.0214176
time: 4.00 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0015581, 0.0091118, -0.0023114, 0.0091812, -0.0107392, 0.0114232
1: -0.0053192, 0.0057308, -0.0060369, 0.0058652, -0.0111844, 0.0117677
2: -0.0385326, 0.0165435, -0.0403286, 0.0178638, -0.0563963, 0.0568721
3: -0.0039902, 0.0190617, -0.0041081, 0.0206202, -0.0246104, 0.0231697
4: 0.0019767, 0.0218408, 0.0014046, 0.0223105, -0.0203338, 0.0204362
5: -0.0037801, 0.0255936, -0.0038502, 0.0277883, -0.0315684, 0.0294438
6: 0.9917475, 1.0146971, 0.9915908, 1.0161757, -0.0244282, 0.0231063
7: -0.0098047, 0.0261527, -0.0108404, 0.0270029, -0.0345833, 0.0352804
8: -0.0029019, 0.0091818, -0.0035673, 0.0094482, -0.0123501, 0.0127491
9: -0.0383281, -0.0031709, -0.0398369, -0.0025234, -0.0358048, 0.0366660

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=71, inp2_unstable=78, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 188

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 92

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0214011, upper bound: 0.0213962
time: 3.53 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213949, upper bound: 0.0213949
time: 3.36 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0015910, 0.0091148, -0.0034763, 0.0092884, -0.0108794, 0.0125911
1: -0.0053506, 0.0057367, -0.0071466, 0.0060730, -0.0114236, 0.0128833
2: -0.0386111, 0.0168115, -0.0431059, 0.0164736, -0.0550847, 0.0599174
3: -0.0040141, 0.0191298, -0.0039839, 0.0230303, -0.0270444, 0.0231137
4: 0.0018606, 0.0218613, 0.0020070, 0.0230368, -0.0211762, 0.0198544
5: -0.0037832, 0.0256896, -0.0039586, 0.0311822, -0.0349654, 0.0296482
6: 0.9917157, 1.0147618, 0.9917558, 1.0184627, -0.0267470, 0.0230060
7: -0.0100149, 0.0261899, -0.0097499, 0.0283176, -0.0365300, 0.0338916
8: -0.0029310, 0.0091935, -0.0045962, 0.0098601, -0.0127911, 0.0137897
9: -0.0383941, -0.0030395, -0.0421701, -0.0032052, -0.0351889, 0.0391306

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=78, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 188

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0205410, upper bound: 0.0206296
time: 2.64 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0205150, upper bound: 0.0206022
time: 3.74 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0015762, 0.0091135, -0.0039345, 0.0093307, -0.0109069, 0.0130480
1: -0.0053366, 0.0057340, -0.0075832, 0.0061547, -0.0114913, 0.0133172
2: -0.0385759, 0.0165291, -0.0441985, 0.0164607, -0.0550367, 0.0607277
3: -0.0039889, 0.0190993, -0.0039828, 0.0239784, -0.0279673, 0.0230821
4: 0.0019829, 0.0218521, 0.0020126, 0.0233225, -0.0213396, 0.0198396
5: -0.0037818, 0.0256466, -0.0040013, 0.0325174, -0.0362992, 0.0296479
6: 0.9917492, 1.0147327, 0.9917573, 1.0193622, -0.0276130, 0.0229754
7: -0.0097934, 0.0261733, -0.0097398, 0.0288348, -0.0369484, 0.0340095
8: -0.0029180, 0.0091882, -0.0050009, 0.0100221, -0.0129401, 0.0141891
9: -0.0383646, -0.0031780, -0.0430879, -0.0032116, -0.0351530, 0.0399099

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=69, inp2_unstable=77, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 188

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0205392, upper bound: 0.0206315
time: 4.11 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0205125, upper bound: 0.0206032
time: 3.86 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 9.34 seconds
IS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 9.34
Output dim: 6, lower bound: -0.0208814, upper bound: 0.0203548
IS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 9.34
Output dim: 6, lower bound: -0.0208272, upper bound: 0.0202831
IS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 9.34
Output dim: 6, lower bound: -0.0208814, upper bound: 0.0203423
IS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 9.34
Output dim: 6, lower bound: -0.0208266, upper bound: 0.0202674
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 9.34
Output dim: 6, lower bound: -0.0214082, upper bound: 0.0214180
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 9.34
Output dim: 6, lower bound: -0.0214038, upper bound: 0.0214176
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 9.34
Output dim: 6, lower bound: -0.0214011, upper bound: 0.0213962
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 9.34
Output dim: 6, lower bound: -0.0213949, upper bound: 0.0213949
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 9.34
Output dim: 6, lower bound: -0.0205410, upper bound: 0.0206296
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 9.34
Output dim: 6, lower bound: -0.0205150, upper bound: 0.0206022
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 9.34
Output dim: 6, lower bound: -0.0205392, upper bound: 0.0206315
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 9.34
Output dim: 6, lower bound: -0.0205125, upper bound: 0.0206032

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0011441, 0.0090737, -0.0014750, 0.0091041, -0.0102482, 0.0105486
1: -0.0049249, 0.0056570, -0.0052401, 0.0057160, -0.0106409, 0.0108970
2: -0.0375457, 0.0155242, -0.0383345, 0.0162973, -0.0538429, 0.0538586
3: -0.0038991, 0.0182053, -0.0039682, 0.0188898, -0.0227889, 0.0221734
4: 0.0024184, 0.0215827, 0.0020834, 0.0217890, -0.0193706, 0.0194993
5: -0.0037416, 0.0243876, -0.0037724, 0.0253515, -0.0290931, 0.0281599
6: 0.9918685, 1.0138845, 0.9917767, 1.0145340, -0.0226655, 0.0221078
7: -0.0090051, 0.0256856, -0.0096116, 0.0260590, -0.0327086, 0.0329824
8: -0.0025363, 0.0090355, -0.0028285, 0.0091524, -0.0116887, 0.0118640
9: -0.0374991, -0.0036709, -0.0381617, -0.0032917, -0.0342073, 0.0344908

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=35, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=69, inp2_unstable=71, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 188

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0200521, upper bound: 0.0197764
time: 3.64 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0208487, upper bound: 0.0203238
time: 3.58 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0015870, 0.0091145, -0.0014594, 0.0091027, -0.0106897, 0.0105739
1: -0.0053468, 0.0057360, -0.0052253, 0.0057132, -0.0110600, 0.0109613
2: -0.0386015, 0.0155544, -0.0382975, 0.0159985, -0.0546000, 0.0538519
3: -0.0039018, 0.0191215, -0.0039415, 0.0188577, -0.0227595, 0.0230629
4: 0.0024053, 0.0218588, 0.0022128, 0.0217793, -0.0193740, 0.0196460
5: -0.0037828, 0.0256778, -0.0037709, 0.0253064, -0.0290892, 0.0294487
6: 0.9918649, 1.0147538, 0.9918122, 1.0145035, -0.0226386, 0.0229416
7: -0.0090288, 0.0261854, -0.0093772, 0.0260415, -0.0328675, 0.0334123
8: -0.0029275, 0.0091920, -0.0028149, 0.0091470, -0.0120744, 0.0120069
9: -0.0383860, -0.0036561, -0.0381307, -0.0034383, -0.0349478, 0.0344746

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=35, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=71, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 188

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0200013, upper bound: 0.0197303
time: 4.05 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0207950, upper bound: 0.0202521
time: 3.52 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0011097, 0.0090705, -0.0021139, 0.0091630, -0.0102727, 0.0111844
1: -0.0048921, 0.0056508, -0.0058488, 0.0058300, -0.0107221, 0.0114996
2: -0.0374636, 0.0150504, -0.0398578, 0.0174173, -0.0548809, 0.0549083
3: -0.0038568, 0.0181341, -0.0040682, 0.0202117, -0.0240685, 0.0222023
4: 0.0026237, 0.0215613, 0.0015980, 0.0221874, -0.0195637, 0.0199632
5: -0.0037384, 0.0242874, -0.0038318, 0.0272131, -0.0309515, 0.0281192
6: 0.9919247, 1.0138171, 0.9916438, 1.0157883, -0.0238636, 0.0221732
7: -0.0086335, 0.0256467, -0.0104901, 0.0267801, -0.0331434, 0.0349687
8: -0.0025060, 0.0090233, -0.0033929, 0.0093784, -0.0118843, 0.0124161
9: -0.0374302, -0.0039033, -0.0394415, -0.0027424, -0.0346878, 0.0355382

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=35, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=69, inp2_unstable=76, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 193

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0203239, upper bound: 0.0198688
time: 3.12 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0203074, upper bound: 0.0197802
time: 2.98 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0015529, 0.0091113, -0.0020985, 0.0091616, -0.0107144, 0.0112099
1: -0.0053143, 0.0057299, -0.0058341, 0.0058272, -0.0111415, 0.0115640
2: -0.0385202, 0.0150839, -0.0398212, 0.0171076, -0.0556279, 0.0549051
3: -0.0038598, 0.0190510, -0.0040405, 0.0201799, -0.0240397, 0.0230915
4: 0.0026092, 0.0218376, 0.0017322, 0.0221778, -0.0195686, 0.0201053
5: -0.0037796, 0.0255785, -0.0038304, 0.0271683, -0.0309479, 0.0294089
6: 0.9919207, 1.0146868, 0.9916806, 1.0157580, -0.0238374, 0.0230063
7: -0.0086598, 0.0261469, -0.0102472, 0.0267627, -0.0333119, 0.0353793
8: -0.0028973, 0.0091800, -0.0033793, 0.0093729, -0.0122702, 0.0125593
9: -0.0383178, -0.0038869, -0.0394106, -0.0028942, -0.0354235, 0.0355238

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=35, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 193

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0202363, upper bound: 0.0197232
time: 3.12 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0202167, upper bound: 0.0196503
time: 3.40 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0015489, 0.0091109, -0.0016720, 0.0091223, -0.0106712, 0.0107830
1: -0.0053105, 0.0057292, -0.0054278, 0.0057511, -0.0110616, 0.0111570
2: -0.0385108, 0.0159828, -0.0388042, 0.0163864, -0.0548971, 0.0547870
3: -0.0039401, 0.0190427, -0.0039761, 0.0192974, -0.0232375, 0.0230189
4: 0.0022197, 0.0218351, 0.0020448, 0.0219118, -0.0196922, 0.0197903
5: -0.0037793, 0.0255669, -0.0037907, 0.0259256, -0.0297049, 0.0293576
6: 0.9918140, 1.0146792, 0.9917661, 1.0149208, -0.0231068, 0.0229131
7: -0.0093648, 0.0261424, -0.0096815, 0.0262813, -0.0333177, 0.0329949
8: -0.0028939, 0.0091786, -0.0030026, 0.0092221, -0.0121160, 0.0121811
9: -0.0383098, -0.0034460, -0.0385564, -0.0032480, -0.0350618, 0.0351104

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=35, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=69, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 188

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0203999, upper bound: 0.0206690
time: 3.08 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213769, upper bound: 0.0213871
time: 3.73 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0019443, 0.0091474, -0.0016569, 0.0091209, -0.0110652, 0.0108043
1: -0.0056872, 0.0057997, -0.0054134, 0.0057484, -0.0114357, 0.0112131
2: -0.0394536, 0.0159660, -0.0387683, 0.0160896, -0.0555432, 0.0547343
3: -0.0039386, 0.0198609, -0.0039496, 0.0192662, -0.0232048, 0.0238105
4: 0.0022269, 0.0220816, 0.0021734, 0.0219024, -0.0196755, 0.0199083
5: -0.0038161, 0.0267190, -0.0037893, 0.0258817, -0.0296977, 0.0305083
6: 0.9918160, 1.0154554, 0.9918013, 1.0148910, -0.0230750, 0.0236540
7: -0.0093517, 0.0265887, -0.0094487, 0.0262643, -0.0334469, 0.0334412
8: -0.0032431, 0.0093184, -0.0029892, 0.0092168, -0.0124599, 0.0123076
9: -0.0391018, -0.0034542, -0.0385262, -0.0033936, -0.0357083, 0.0350719

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=35, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 188

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0203959, upper bound: 0.0206690
time: 3.25 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213731, upper bound: 0.0213868
time: 3.57 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0015142, 0.0091078, -0.0022976, 0.0091799, -0.0106941, 0.0114054
1: -0.0052775, 0.0057230, -0.0060238, 0.0058627, -0.0111402, 0.0117468
2: -0.0384281, 0.0154889, -0.0402958, 0.0175060, -0.0559341, 0.0557847
3: -0.0038960, 0.0189710, -0.0040761, 0.0205918, -0.0244877, 0.0230471
4: 0.0024337, 0.0218135, 0.0015596, 0.0223019, -0.0198682, 0.0202539
5: -0.0037760, 0.0254659, -0.0038489, 0.0277483, -0.0315243, 0.0293148
6: 0.9918726, 1.0146110, 0.9916334, 1.0161489, -0.0242763, 0.0229777
7: -0.0089775, 0.0261033, -0.0105597, 0.0269874, -0.0336912, 0.0349335
8: -0.0028632, 0.0091663, -0.0035551, 0.0094433, -0.0123066, 0.0127214
9: -0.0382404, -0.0036882, -0.0398094, -0.0026989, -0.0355415, 0.0361212

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=35, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=69, inp2_unstable=77, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 193

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0209223, upper bound: 0.0209411
time: 3.67 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0208659, upper bound: 0.0208608
time: 3.22 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0019102, 0.0091442, -0.0022826, 0.0091785, -0.0110888, 0.0114269
1: -0.0056547, 0.0057936, -0.0060095, 0.0058600, -0.0115148, 0.0118031
2: -0.0393722, 0.0154693, -0.0402601, 0.0171980, -0.0565702, 0.0557294
3: -0.0038942, 0.0197903, -0.0040486, 0.0205608, -0.0244550, 0.0238389
4: 0.0024422, 0.0220604, 0.0016931, 0.0222925, -0.0198504, 0.0203673
5: -0.0038129, 0.0266196, -0.0038475, 0.0277046, -0.0315175, 0.0304672
6: 0.9918749, 1.0153884, 0.9916698, 1.0161195, -0.0242445, 0.0237185
7: -0.0089621, 0.0265502, -0.0103181, 0.0269705, -0.0338287, 0.0353497
8: -0.0032130, 0.0093063, -0.0035419, 0.0094380, -0.0126510, 0.0128482
9: -0.0390335, -0.0036978, -0.0397794, -0.0028499, -0.0361836, 0.0360816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=35, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 193

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0209046, upper bound: 0.0209234
time: 2.90 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0208507, upper bound: 0.0208507
time: 3.19 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 7.47 seconds
IS_A1_B1_A1_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 7.47
Output dim: 6, lower bound: -0.0200521, upper bound: 0.0197764
IS_A1_B1_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 7.47
Output dim: 6, lower bound: -0.0208487, upper bound: 0.0203238
IS_A1_B1_A1_B1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 7.47
Output dim: 6, lower bound: -0.0200013, upper bound: 0.0197303
IS_A1_B1_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 7.47
Output dim: 6, lower bound: -0.0207950, upper bound: 0.0202521
IS_A1_B1_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 7.47
Output dim: 6, lower bound: -0.0203239, upper bound: 0.0198688
IS_A1_B1_A1_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 7.47
Output dim: 6, lower bound: -0.0203074, upper bound: 0.0197802
IS_A1_B1_A1_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 7.47
Output dim: 6, lower bound: -0.0202363, upper bound: 0.0197232
IS_A1_B1_A1_B1_A1_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 7.47
Output dim: 6, lower bound: -0.0202167, upper bound: 0.0196503
IS_A1_B1_A1_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 7.47
Output dim: 6, lower bound: -0.0203999, upper bound: 0.0206690
IS_A1_B1_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 7.47
Output dim: 6, lower bound: -0.0213769, upper bound: 0.0213871
IS_A1_B1_A1_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 7.47
Output dim: 6, lower bound: -0.0203959, upper bound: 0.0206690
IS_A1_B1_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 7.47
Output dim: 6, lower bound: -0.0213731, upper bound: 0.0213868
IS_A1_B1_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 7.47
Output dim: 6, lower bound: -0.0209223, upper bound: 0.0209411
IS_A1_B1_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 7.47
Output dim: 6, lower bound: -0.0208659, upper bound: 0.0208608
IS_A1_B1_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 7.47
Output dim: 6, lower bound: -0.0209046, upper bound: 0.0209234
IS_A1_B1_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 7.47
Output dim: 6, lower bound: -0.0208507, upper bound: 0.0208507

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0011441, 0.0090737, -0.0014429, 0.0091012, -0.0102452, 0.0105165
1: -0.0049249, 0.0056570, -0.0052095, 0.0057102, -0.0106351, 0.0108664
2: -0.0375456, 0.0155238, -0.0382579, 0.0157587, -0.0533043, 0.0537817
3: -0.0038991, 0.0182052, -0.0039201, 0.0188233, -0.0227224, 0.0221253
4: 0.0024185, 0.0215827, 0.0023167, 0.0217690, -0.0193504, 0.0192659
5: -0.0037416, 0.0243875, -0.0037694, 0.0252579, -0.0289995, 0.0281569
6: 0.9918684, 1.0138845, 0.9918407, 1.0144708, -0.0226024, 0.0220439
7: -0.0090049, 0.0256855, -0.0091891, 0.0260227, -0.0326728, 0.0326008
8: -0.0025363, 0.0090354, -0.0028001, 0.0091411, -0.0116773, 0.0118356
9: -0.0374990, -0.0036711, -0.0380974, -0.0035559, -0.0339431, 0.0344263

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=69, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 193

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0208487, upper bound: 0.0203238
time: 3.34 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0208487, upper bound: 0.0203238
time: 3.82 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0015869, 0.0091144, -0.0014272, 0.0090997, -0.0106867, 0.0105416
1: -0.0053467, 0.0057359, -0.0051946, 0.0057075, -0.0110542, 0.0109305
2: -0.0386014, 0.0155540, -0.0382206, 0.0154684, -0.0540698, 0.0537746
3: -0.0039018, 0.0191214, -0.0038941, 0.0187910, -0.0226927, 0.0230156
4: 0.0024054, 0.0218588, 0.0024426, 0.0217592, -0.0193538, 0.0194162
5: -0.0037828, 0.0256777, -0.0037679, 0.0252124, -0.0289952, 0.0294457
6: 0.9918650, 1.0147537, 0.9918751, 1.0144402, -0.0225752, 0.0228786
7: -0.0090286, 0.0261853, -0.0089614, 0.0260051, -0.0328315, 0.0329986
8: -0.0029274, 0.0091920, -0.0027864, 0.0091355, -0.0120630, 0.0119784
9: -0.0383860, -0.0036563, -0.0380661, -0.0036983, -0.0346877, 0.0344098

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 193

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0207950, upper bound: 0.0202521
time: 4.06 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0207950, upper bound: 0.0202521
time: 4.01 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0015489, 0.0091109, -0.0016401, 0.0091194, -0.0106682, 0.0107511
1: -0.0053105, 0.0057292, -0.0053974, 0.0057454, -0.0110559, 0.0111266
2: -0.0385107, 0.0159824, -0.0387283, 0.0158431, -0.0543538, 0.0547107
3: -0.0039400, 0.0190427, -0.0039276, 0.0192315, -0.0231716, 0.0229703
4: 0.0022198, 0.0218351, 0.0022802, 0.0218920, -0.0196722, 0.0195549
5: -0.0037793, 0.0255669, -0.0037877, 0.0258328, -0.0296120, 0.0293546
6: 0.9918140, 1.0146791, 0.9918306, 1.0148581, -0.0230442, 0.0228484
7: -0.0093646, 0.0261424, -0.0092553, 0.0262454, -0.0332819, 0.0325893
8: -0.0028938, 0.0091786, -0.0029744, 0.0092108, -0.0121047, 0.0121530
9: -0.0383098, -0.0034462, -0.0384926, -0.0035145, -0.0347953, 0.0350464

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=69, inp2_unstable=71, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 188

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213768, upper bound: 0.0213870
time: 5.00 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213768, upper bound: 0.0213870
time: 3.97 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0019443, 0.0091474, -0.0016248, 0.0091179, -0.0110623, 0.0107722
1: -0.0056872, 0.0057997, -0.0053829, 0.0057427, -0.0114299, 0.0111826
2: -0.0394535, 0.0159656, -0.0386918, 0.0155521, -0.0550056, 0.0546575
3: -0.0039385, 0.0198608, -0.0039016, 0.0191999, -0.0231384, 0.0237624
4: 0.0022271, 0.0220816, 0.0024063, 0.0218824, -0.0196554, 0.0196753
5: -0.0038161, 0.0267190, -0.0037863, 0.0257883, -0.0296043, 0.0305053
6: 0.9918160, 1.0154552, 0.9918652, 1.0148282, -0.0230122, 0.0235901
7: -0.0093514, 0.0265887, -0.0090270, 0.0262281, -0.0334109, 0.0330068
8: -0.0032431, 0.0093184, -0.0029609, 0.0092054, -0.0124486, 0.0122793
9: -0.0391018, -0.0034544, -0.0384619, -0.0036572, -0.0354446, 0.0350076

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 193

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213731, upper bound: 0.0213868
time: 4.00 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0213731, upper bound: 0.0213868
time: 3.75 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0014832, 0.0091049, -0.0020668, 0.0091586, -0.0106419, 0.0111717
1: -0.0052479, 0.0057174, -0.0058040, 0.0058216, -0.0110695, 0.0115214
2: -0.0383541, 0.0154201, -0.0397457, 0.0170148, -0.0553690, 0.0551658
3: -0.0038898, 0.0189068, -0.0040323, 0.0201144, -0.0240042, 0.0229391
4: 0.0024635, 0.0217941, 0.0017724, 0.0221580, -0.0196945, 0.0200217
5: -0.0037731, 0.0253755, -0.0038275, 0.0270760, -0.0308492, 0.0292030
6: 0.9918808, 1.0145501, 0.9916916, 1.0156958, -0.0238150, 0.0228585
7: -0.0089235, 0.0260683, -0.0101745, 0.0267270, -0.0333816, 0.0345092
8: -0.0028358, 0.0091554, -0.0033513, 0.0093617, -0.0121975, 0.0125067
9: -0.0381782, -0.0037220, -0.0393472, -0.0029398, -0.0352385, 0.0356253

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=69, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 188

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0195891, upper bound: 0.0193369
time: 2.93 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0208929, upper bound: 0.0209109
time: 2.90 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0013419, 0.0090919, -0.0020037, 0.0091528, -0.0104947, 0.0110956
1: -0.0051133, 0.0056922, -0.0057438, 0.0058103, -0.0109236, 0.0114361
2: -0.0380172, 0.0150949, -0.0395952, 0.0187098, -0.0567270, 0.0546901
3: -0.0038608, 0.0186144, -0.0041836, 0.0199838, -0.0238445, 0.0227981
4: 0.0026044, 0.0217060, 0.0010379, 0.0221187, -0.0195142, 0.0206681
5: -0.0037600, 0.0249638, -0.0038216, 0.0268921, -0.0306521, 0.0287854
6: 0.9919193, 1.0142727, 0.9914905, 1.0155721, -0.0236527, 0.0227822
7: -0.0086684, 0.0259088, -0.0115040, 0.0266558, -0.0332725, 0.0354898
8: -0.0027110, 0.0091054, -0.0032956, 0.0093394, -0.0120504, 0.0124010
9: -0.0378952, -0.0038815, -0.0392208, -0.0021084, -0.0357867, 0.0353393

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=68, inp2_unstable=76, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 188

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0195417, upper bound: 0.0192413
time: 2.88 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0208366, upper bound: 0.0208318
time: 3.99 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0018785, 0.0091413, -0.0020519, 0.0091573, -0.0110358, 0.0111932
1: -0.0056245, 0.0057880, -0.0057897, 0.0058189, -0.0114434, 0.0115776
2: -0.0392966, 0.0154004, -0.0397099, 0.0167074, -0.0560041, 0.0551103
3: -0.0038881, 0.0197247, -0.0040048, 0.0200834, -0.0239714, 0.0237295
4: 0.0024720, 0.0220406, 0.0019056, 0.0221487, -0.0196766, 0.0201349
5: -0.0038099, 0.0265273, -0.0038261, 0.0270324, -0.0308423, 0.0303534
6: 0.9918831, 1.0153261, 0.9917281, 1.0156665, -0.0237834, 0.0235980
7: -0.0089080, 0.0265144, -0.0099333, 0.0267101, -0.0335184, 0.0349266
8: -0.0031850, 0.0092951, -0.0033381, 0.0093564, -0.0125414, 0.0126333
9: -0.0389700, -0.0037316, -0.0393172, -0.0030905, -0.0358795, 0.0355856

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=74, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 193

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0194236, upper bound: 0.0190478
time: 2.97 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0208759, upper bound: 0.0208938
time: 3.29 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0017441, 0.0091289, -0.0019888, 0.0091515, -0.0108956, 0.0111178
1: -0.0054965, 0.0057640, -0.0057296, 0.0058076, -0.0113042, 0.0114936
2: -0.0389763, 0.0150689, -0.0395597, 0.0184177, -0.0573940, 0.0546285
3: -0.0038584, 0.0194467, -0.0041575, 0.0199529, -0.0238114, 0.0236042
4: 0.0026157, 0.0219568, 0.0011645, 0.0221094, -0.0194937, 0.0207923
5: -0.0037974, 0.0261358, -0.0038202, 0.0268487, -0.0306461, 0.0299560
6: 0.9919225, 1.0150623, 0.9915251, 1.0155426, -0.0236201, 0.0235373
7: -0.0086480, 0.0263628, -0.0112749, 0.0266389, -0.0334015, 0.0359291
8: -0.0030663, 0.0092476, -0.0032824, 0.0093341, -0.0124004, 0.0125301
9: -0.0387009, -0.0038942, -0.0391910, -0.0022517, -0.0364492, 0.0352967

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 193

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0193792, upper bound: 0.0189736
time: 2.83 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0208218, upper bound: 0.0208218
time: 3.86 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 8.06 seconds
IS_A1_B1_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 8.06
Output dim: 6, lower bound: -0.0208487, upper bound: 0.0203238
IS_A1_B1_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 8.06
Output dim: 6, lower bound: -0.0208487, upper bound: 0.0203238
IS_A1_B1_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 8.06
Output dim: 6, lower bound: -0.0207950, upper bound: 0.0202521
IS_A1_B1_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 8.06
Output dim: 6, lower bound: -0.0207950, upper bound: 0.0202521
IS_A1_B1_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 8.06
Output dim: 6, lower bound: -0.0213768, upper bound: 0.0213870
IS_A1_B1_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 8.06
Output dim: 6, lower bound: -0.0213768, upper bound: 0.0213870
IS_A1_B1_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 8.06
Output dim: 6, lower bound: -0.0213731, upper bound: 0.0213868
IS_A1_B1_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 8.06
Output dim: 6, lower bound: -0.0213731, upper bound: 0.0213868
IS_A1_B1_A1_B1_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 9, time: 8.06
Output dim: 6, lower bound: -0.0195891, upper bound: 0.0193369
IS_A1_B1_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 8.06
Output dim: 6, lower bound: -0.0208929, upper bound: 0.0209109
IS_A1_B1_A1_B1_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 9, time: 8.06
Output dim: 6, lower bound: -0.0195417, upper bound: 0.0192413
IS_A1_B1_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 8.06
Output dim: 6, lower bound: -0.0208366, upper bound: 0.0208318
IS_A1_B1_A1_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 8.06
Output dim: 6, lower bound: -0.0194236, upper bound: 0.0190478
IS_A1_B1_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 8.06
Output dim: 6, lower bound: -0.0208759, upper bound: 0.0208938
IS_A1_B1_A1_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 8.06
Output dim: 6, lower bound: -0.0193792, upper bound: 0.0189736
IS_A1_B1_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 8.06
Output dim: 6, lower bound: -0.0208218, upper bound: 0.0208218

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0010980, 0.0090694, -0.0014429, 0.0091012, -0.0101991, 0.0105123
1: -0.0048809, 0.0056487, -0.0052095, 0.0057102, -0.0105912, 0.0108582
2: -0.0374357, 0.0150335, -0.0382579, 0.0157587, -0.0531944, 0.0532914
3: -0.0038553, 0.0181098, -0.0039201, 0.0188233, -0.0226786, 0.0220299
4: 0.0026310, 0.0215539, 0.0023167, 0.0217690, -0.0191379, 0.0192372
5: -0.0037373, 0.0242532, -0.0037694, 0.0252579, -0.0289952, 0.0280226
6: 0.9919267, 1.0137939, 0.9918407, 1.0144708, -0.0225441, 0.0219533
7: -0.0086202, 0.0256335, -0.0091891, 0.0260227, -0.0322006, 0.0324904
8: -0.0024956, 0.0090191, -0.0028001, 0.0091411, -0.0116367, 0.0118193
9: -0.0374067, -0.0039116, -0.0380974, -0.0035559, -0.0338508, 0.0341858

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=68, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 193

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0202949, upper bound: 0.0198571
time: 3.18 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0202820, upper bound: 0.0197904
time: 3.09 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0017458, 0.0091291, -0.0014429, 0.0091012, -0.0108469, 0.0105719
1: -0.0054981, 0.0057643, -0.0052095, 0.0057102, -0.0112083, 0.0109738
2: -0.0389801, 0.0159540, -0.0382579, 0.0157587, -0.0547388, 0.0542119
3: -0.0039375, 0.0194501, -0.0039201, 0.0188233, -0.0227608, 0.0233701
4: 0.0022321, 0.0219578, 0.0023167, 0.0217690, -0.0195368, 0.0196411
5: -0.0037976, 0.0261405, -0.0037694, 0.0252579, -0.0290555, 0.0299099
6: 0.9918174, 1.0150656, 0.9918407, 1.0144708, -0.0226534, 0.0232249
7: -0.0093423, 0.0263646, -0.0091891, 0.0260227, -0.0330584, 0.0334315
8: -0.0030677, 0.0092482, -0.0028001, 0.0091411, -0.0122088, 0.0120483
9: -0.0387041, -0.0034601, -0.0380974, -0.0035559, -0.0351482, 0.0346373

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 193

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0202949, upper bound: 0.0198571
time: 3.05 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0202820, upper bound: 0.0197904
time: 3.10 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0015398, 0.0091101, -0.0014272, 0.0090997, -0.0106396, 0.0105373
1: -0.0053019, 0.0057276, -0.0051946, 0.0057075, -0.0110094, 0.0109221
2: -0.0384892, 0.0150609, -0.0382206, 0.0154684, -0.0539576, 0.0532815
3: -0.0038577, 0.0190240, -0.0038941, 0.0187910, -0.0226487, 0.0229182
4: 0.0026191, 0.0218294, 0.0024426, 0.0217592, -0.0191401, 0.0193869
5: -0.0037784, 0.0255406, -0.0037679, 0.0252124, -0.0289908, 0.0293085
6: 0.9919233, 1.0146613, 0.9918751, 1.0144402, -0.0225168, 0.0227862
7: -0.0086418, 0.0261322, -0.0089614, 0.0260051, -0.0323523, 0.0328901
8: -0.0028858, 0.0091754, -0.0027864, 0.0091355, -0.0120214, 0.0119617
9: -0.0382917, -0.0038981, -0.0380661, -0.0036983, -0.0345934, 0.0341679

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 193

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0202123, upper bound: 0.0197155
time: 3.08 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0202018, upper bound: 0.0196630
time: 3.48 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0022028, 0.0091712, -0.0014272, 0.0090997, -0.0113026, 0.0105984
1: -0.0059335, 0.0058458, -0.0051946, 0.0057075, -0.0116410, 0.0110404
2: -0.0400698, 0.0159600, -0.0382206, 0.0154684, -0.0555382, 0.0541806
3: -0.0039380, 0.0203957, -0.0038941, 0.0187910, -0.0227290, 0.0242898
4: 0.0022295, 0.0222428, 0.0024426, 0.0217592, -0.0195297, 0.0198002
5: -0.0038401, 0.0274722, -0.0037679, 0.0252124, -0.0290525, 0.0312401
6: 0.9918167, 1.0159628, 0.9918751, 1.0144402, -0.0226235, 0.0240877
7: -0.0093470, 0.0268804, -0.0089614, 0.0260051, -0.0331854, 0.0338492
8: -0.0034714, 0.0094098, -0.0027864, 0.0091355, -0.0126070, 0.0121962
9: -0.0396196, -0.0034572, -0.0380661, -0.0036983, -0.0359213, 0.0346089

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=74, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 193

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0202123, upper bound: 0.0197155
time: 3.46 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0202018, upper bound: 0.0196630
time: 3.18 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0015016, 0.0091066, -0.0016401, 0.0091194, -0.0106209, 0.0107467
1: -0.0052655, 0.0057207, -0.0053974, 0.0057454, -0.0110109, 0.0111182
2: -0.0383980, 0.0154989, -0.0387283, 0.0158431, -0.0542411, 0.0542272
3: -0.0038969, 0.0189449, -0.0039276, 0.0192315, -0.0231284, 0.0228725
4: 0.0024294, 0.0218056, 0.0022802, 0.0218920, -0.0194626, 0.0195254
5: -0.0037749, 0.0254292, -0.0037877, 0.0258328, -0.0296076, 0.0292169
6: 0.9918714, 1.0145863, 0.9918306, 1.0148581, -0.0229867, 0.0227557
7: -0.0089853, 0.0260890, -0.0092553, 0.0262454, -0.0328218, 0.0324772
8: -0.0028521, 0.0091619, -0.0029744, 0.0092108, -0.0120629, 0.0121362
9: -0.0382151, -0.0036833, -0.0384926, -0.0035145, -0.0347006, 0.0348092

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=68, inp2_unstable=71, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 193

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0209074, upper bound: 0.0209349
time: 3.76 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0208703, upper bound: 0.0208771
time: 3.29 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0021130, 0.0091629, -0.0016401, 0.0091194, -0.0112323, 0.0108030
1: -0.0058479, 0.0058298, -0.0053974, 0.0057454, -0.0115933, 0.0112272
2: -0.0398556, 0.0164486, -0.0387283, 0.0158431, -0.0556987, 0.0551769
3: -0.0039817, 0.0202097, -0.0039276, 0.0192315, -0.0232132, 0.0241373
4: 0.0020178, 0.0221868, 0.0022802, 0.0218920, -0.0198742, 0.0199066
5: -0.0038318, 0.0272103, -0.0037877, 0.0258328, -0.0296645, 0.0309981
6: 0.9917587, 1.0157863, 0.9918306, 1.0148581, -0.0230994, 0.0239556
7: -0.0097303, 0.0267790, -0.0092553, 0.0262454, -0.0336778, 0.0334184
8: -0.0033921, 0.0093780, -0.0029744, 0.0092108, -0.0126029, 0.0123524
9: -0.0394396, -0.0032175, -0.0384926, -0.0035145, -0.0359251, 0.0352751

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=71, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 193

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0209074, upper bound: 0.0209349
time: 3.67 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0208703, upper bound: 0.0208771
time: 3.39 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0018966, 0.0091430, -0.0016248, 0.0091179, -0.0110145, 0.0107678
1: -0.0056417, 0.0057912, -0.0053829, 0.0057427, -0.0113845, 0.0111741
2: -0.0393397, 0.0154797, -0.0386918, 0.0155521, -0.0548918, 0.0541716
3: -0.0038951, 0.0197621, -0.0039016, 0.0191999, -0.0230950, 0.0236637
4: 0.0024377, 0.0220519, 0.0024063, 0.0218824, -0.0194448, 0.0196456
5: -0.0038116, 0.0265799, -0.0037863, 0.0257883, -0.0295999, 0.0303662
6: 0.9918737, 1.0153617, 0.9918652, 1.0148282, -0.0229545, 0.0234965
7: -0.0089703, 0.0265348, -0.0090270, 0.0262281, -0.0329435, 0.0328967
8: -0.0032009, 0.0093015, -0.0029609, 0.0092054, -0.0124063, 0.0122624
9: -0.0390062, -0.0036927, -0.0384619, -0.0036572, -0.0353489, 0.0347692

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 188

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0208919, upper bound: 0.0209238
time: 3.65 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0208577, upper bound: 0.0208699
time: 3.64 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0025351, 0.0092018, -0.0016248, 0.0091179, -0.0116531, 0.0108266
1: -0.0062500, 0.0059051, -0.0053829, 0.0057427, -0.0119928, 0.0112880
2: -0.0408620, 0.0163941, -0.0386918, 0.0155521, -0.0564141, 0.0550859
3: -0.0039768, 0.0210831, -0.0039016, 0.0191999, -0.0231767, 0.0249847
4: 0.0020414, 0.0224500, 0.0024063, 0.0218824, -0.0198410, 0.0200437
5: -0.0038710, 0.0284403, -0.0037863, 0.0257883, -0.0296593, 0.0322266
6: 0.9917652, 1.0166150, 0.9918652, 1.0148282, -0.0230630, 0.0247499
7: -0.0096875, 0.0272554, -0.0090270, 0.0262281, -0.0337729, 0.0338589
8: -0.0037649, 0.0095273, -0.0029609, 0.0092054, -0.0129704, 0.0124882
9: -0.0402851, -0.0032442, -0.0384619, -0.0036572, -0.0366279, 0.0352177

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 188

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0208919, upper bound: 0.0209238
time: 3.59 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0208577, upper bound: 0.0208699
time: 3.89 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0014516, 0.0091020, -0.0020669, 0.0091586, -0.0106102, 0.0111688
1: -0.0052178, 0.0057118, -0.0058039, 0.0058216, -0.0110394, 0.0115157
2: -0.0382787, 0.0148868, -0.0397456, 0.0170145, -0.0552933, 0.0546324
3: -0.0038422, 0.0188414, -0.0040322, 0.0201143, -0.0239565, 0.0228736
4: 0.0026946, 0.0217744, 0.0017726, 0.0221580, -0.0194634, 0.0200018
5: -0.0037702, 0.0252834, -0.0038275, 0.0270759, -0.0308461, 0.0291108
6: 0.9919441, 1.0144880, 0.9916916, 1.0156959, -0.0237519, 0.0227963
7: -0.0085052, 0.0260326, -0.0101742, 0.0267270, -0.0330368, 0.0344737
8: -0.0028079, 0.0091442, -0.0033513, 0.0093617, -0.0121696, 0.0124954
9: -0.0381149, -0.0039835, -0.0393472, -0.0029399, -0.0351750, 0.0353636

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=69, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 193

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0194777, upper bound: 0.0197772
time: 3.12 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0194777, upper bound: 0.0209109
time: 3.13 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0013102, 0.0090890, -0.0020037, 0.0091528, -0.0104631, 0.0110927
1: -0.0050832, 0.0056866, -0.0057438, 0.0058103, -0.0108934, 0.0114304
2: -0.0379417, 0.0145627, -0.0395951, 0.0187094, -0.0566512, 0.0541578
3: -0.0038132, 0.0185490, -0.0041836, 0.0199837, -0.0237970, 0.0227326
4: 0.0028350, 0.0216863, 0.0010381, 0.0221187, -0.0192836, 0.0206482
5: -0.0037570, 0.0248716, -0.0038216, 0.0268921, -0.0306491, 0.0286932
6: 0.9919825, 1.0142106, 0.9914905, 1.0155720, -0.0235894, 0.0227200
7: -0.0082509, 0.0258731, -0.0115037, 0.0266557, -0.0329235, 0.0354545
8: -0.0026831, 0.0090942, -0.0032956, 0.0093394, -0.0120225, 0.0123898
9: -0.0378318, -0.0041425, -0.0392208, -0.0021086, -0.0357232, 0.0350783

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=68, inp2_unstable=76, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 132

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0189897, upper bound: 0.0193835
time: 2.88 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0189897, upper bound: 0.0208318
time: 4.63 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0018467, 0.0091384, -0.0020519, 0.0091573, -0.0110040, 0.0111902
1: -0.0055943, 0.0057823, -0.0057897, 0.0058189, -0.0114131, 0.0115720
2: -0.0392209, 0.0148706, -0.0397099, 0.0167071, -0.0559280, 0.0545805
3: -0.0038407, 0.0196590, -0.0040048, 0.0200833, -0.0239241, 0.0236637
4: 0.0027016, 0.0220208, 0.0019058, 0.0221487, -0.0194471, 0.0201150
5: -0.0038070, 0.0264347, -0.0038261, 0.0270323, -0.0308393, 0.0302608
6: 0.9919460, 1.0152637, 0.9917281, 1.0156665, -0.0237204, 0.0235356
7: -0.0084925, 0.0264786, -0.0099330, 0.0267101, -0.0331556, 0.0348911
8: -0.0031569, 0.0092839, -0.0033381, 0.0093564, -0.0125134, 0.0126220
9: -0.0389064, -0.0039915, -0.0393172, -0.0030907, -0.0358157, 0.0353257

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=74, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 193

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0194521, upper bound: 0.0197637
time: 2.49 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0194521, upper bound: 0.0208938
time: 3.04 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0017126, 0.0091260, -0.0019888, 0.0091515, -0.0108640, 0.0111148
1: -0.0054664, 0.0057584, -0.0057296, 0.0058076, -0.0112741, 0.0114880
2: -0.0389009, 0.0145403, -0.0395596, 0.0184174, -0.0573183, 0.0540998
3: -0.0038112, 0.0193813, -0.0041575, 0.0199529, -0.0237641, 0.0235388
4: 0.0028448, 0.0219371, 0.0011647, 0.0221094, -0.0192646, 0.0207725
5: -0.0037945, 0.0260437, -0.0038202, 0.0268486, -0.0306431, 0.0298639
6: 0.9919851, 1.0150005, 0.9915252, 1.0155426, -0.0235575, 0.0234753
7: -0.0082333, 0.0263271, -0.0112746, 0.0266389, -0.0330361, 0.0358937
8: -0.0030384, 0.0092364, -0.0032824, 0.0093341, -0.0123725, 0.0125189
9: -0.0386376, -0.0041535, -0.0391909, -0.0022518, -0.0363857, 0.0350374

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 132

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0189736, upper bound: 0.0193792
time: 2.86 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0189736, upper bound: 0.0208218
time: 2.63 seconds

## Summary of splitting at layer (split count: 9)
- Time for IS candidates: 6.88 seconds
IS_A1_B1_A1_B1_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 10, time: 6.88
Output dim: 6, lower bound: -0.0202949, upper bound: 0.0198571
IS_A1_B1_A1_B1_A1_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 10, time: 6.88
Output dim: 6, lower bound: -0.0202820, upper bound: 0.0197904
IS_A1_B1_A1_B1_A1_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 10, time: 6.88
Output dim: 6, lower bound: -0.0202949, upper bound: 0.0198571
IS_A1_B1_A1_B1_A1_B1_A1_B2_A2_B2, status: Status.VERIFIED, split count: 10, time: 6.88
Output dim: 6, lower bound: -0.0202820, upper bound: 0.0197904
IS_A1_B1_A1_B1_A1_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 10, time: 6.88
Output dim: 6, lower bound: -0.0202123, upper bound: 0.0197155
IS_A1_B1_A1_B1_A1_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 10, time: 6.88
Output dim: 6, lower bound: -0.0202018, upper bound: 0.0196630
IS_A1_B1_A1_B1_A1_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 10, time: 6.88
Output dim: 6, lower bound: -0.0202123, upper bound: 0.0197155
IS_A1_B1_A1_B1_A1_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 10, time: 6.88
Output dim: 6, lower bound: -0.0202018, upper bound: 0.0196630
IS_A1_B1_A1_B1_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 6.88
Output dim: 6, lower bound: -0.0209074, upper bound: 0.0209349
IS_A1_B1_A1_B1_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 6.88
Output dim: 6, lower bound: -0.0208703, upper bound: 0.0208771
IS_A1_B1_A1_B1_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 6.88
Output dim: 6, lower bound: -0.0209074, upper bound: 0.0209349
IS_A1_B1_A1_B1_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 6.88
Output dim: 6, lower bound: -0.0208703, upper bound: 0.0208771
IS_A1_B1_A1_B1_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 6.88
Output dim: 6, lower bound: -0.0208919, upper bound: 0.0209238
IS_A1_B1_A1_B1_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 6.88
Output dim: 6, lower bound: -0.0208577, upper bound: 0.0208699
IS_A1_B1_A1_B1_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 6.88
Output dim: 6, lower bound: -0.0208919, upper bound: 0.0209238
IS_A1_B1_A1_B1_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 6.88
Output dim: 6, lower bound: -0.0208577, upper bound: 0.0208699
IS_A1_B1_A1_B1_A2_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 10, time: 6.88
Output dim: 6, lower bound: -0.0194777, upper bound: 0.0197772
IS_A1_B1_A1_B1_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 6.88
Output dim: 6, lower bound: -0.0194777, upper bound: 0.0209109
IS_A1_B1_A1_B1_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 10, time: 6.88
Output dim: 6, lower bound: -0.0189897, upper bound: 0.0193835
IS_A1_B1_A1_B1_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 6.88
Output dim: 6, lower bound: -0.0189897, upper bound: 0.0208318
IS_A1_B1_A1_B1_A2_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 10, time: 6.88
Output dim: 6, lower bound: -0.0194521, upper bound: 0.0197637
IS_A1_B1_A1_B1_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 6.88
Output dim: 6, lower bound: -0.0194521, upper bound: 0.0208938
IS_A1_B1_A1_B1_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 10, time: 6.88
Output dim: 6, lower bound: -0.0189736, upper bound: 0.0193792
IS_A1_B1_A1_B1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 6.88
Output dim: 6, lower bound: -0.0189736, upper bound: 0.0208218

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0014706, 0.0091037, -0.0014167, 0.0090988, -0.0105693, 0.0105204
1: -0.0052359, 0.0057152, -0.0051846, 0.0057056, -0.0109415, 0.0108998
2: -0.0383240, 0.0154309, -0.0381956, 0.0153414, -0.0536654, 0.0536266
3: -0.0038908, 0.0188807, -0.0038828, 0.0187693, -0.0226601, 0.0227635
4: 0.0024588, 0.0217863, 0.0024976, 0.0217527, -0.0192939, 0.0192887
5: -0.0037720, 0.0253388, -0.0037670, 0.0251818, -0.0289538, 0.0291058
6: 0.9918795, 1.0145253, 0.9918901, 1.0144196, -0.0225401, 0.0226352
7: -0.0089320, 0.0260540, -0.0088617, 0.0259932, -0.0325296, 0.0320451
8: -0.0028247, 0.0091509, -0.0027771, 0.0091318, -0.0119565, 0.0119280
9: -0.0381530, -0.0037167, -0.0380451, -0.0037606, -0.0343924, 0.0343284

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=68, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 188

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0199117, upper bound: 0.0199992
time: 2.88 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0199117, upper bound: 0.0211564
time: 3.47 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0013296, 0.0090907, -0.0013123, 0.0090892, -0.0104188, 0.0104031
1: -0.0051016, 0.0056900, -0.0050851, 0.0056870, -0.0107885, 0.0107752
2: -0.0379879, 0.0151112, -0.0379467, 0.0170823, -0.0550701, 0.0530579
3: -0.0038622, 0.0185890, -0.0040383, 0.0185533, -0.0224155, 0.0226273
4: 0.0025974, 0.0216983, 0.0017432, 0.0216876, -0.0190902, 0.0199551
5: -0.0037588, 0.0249280, -0.0037572, 0.0248777, -0.0286365, 0.0286852
6: 0.9919174, 1.0142486, 0.9916836, 1.0142148, -0.0222973, 0.0225650
7: -0.0086812, 0.0258949, -0.0102273, 0.0258754, -0.0324171, 0.0331845
8: -0.0027002, 0.0091010, -0.0026849, 0.0090949, -0.0117951, 0.0117860
9: -0.0378705, -0.0038735, -0.0378360, -0.0029067, -0.0349638, 0.0339625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=68, inp2_unstable=69, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 193

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0197716, upper bound: 0.0199346
time: 3.46 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0197716, upper bound: 0.0211061
time: 3.10 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0020807, 0.0091599, -0.0014167, 0.0090988, -0.0111795, 0.0105766
1: -0.0058172, 0.0058240, -0.0051846, 0.0057056, -0.0115228, 0.0110086
2: -0.0397787, 0.0163810, -0.0381956, 0.0153414, -0.0551201, 0.0545766
3: -0.0039756, 0.0201431, -0.0038828, 0.0187693, -0.0227449, 0.0240259
4: 0.0020471, 0.0221667, 0.0024976, 0.0217527, -0.0197056, 0.0196691
5: -0.0038288, 0.0271164, -0.0037670, 0.0251818, -0.0290106, 0.0308834
6: 0.9917668, 1.0157231, 0.9918901, 1.0144196, -0.0226528, 0.0238330
7: -0.0096772, 0.0267426, -0.0088617, 0.0259932, -0.0333866, 0.0329828
8: -0.0033636, 0.0093666, -0.0027771, 0.0091318, -0.0124955, 0.0121437
9: -0.0393750, -0.0032507, -0.0380451, -0.0037606, -0.0356144, 0.0347944

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 193

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0192940, upper bound: 0.0193615
time: 2.83 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0192940, upper bound: 0.0209349
time: 2.73 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0019522, 0.0091481, -0.0013123, 0.0090892, -0.0110414, 0.0104604
1: -0.0056948, 0.0058011, -0.0050851, 0.0056870, -0.0113817, 0.0108863
2: -0.0394724, 0.0160599, -0.0379467, 0.0170823, -0.0565546, 0.0540066
3: -0.0039470, 0.0198772, -0.0040383, 0.0185533, -0.0225003, 0.0239155
4: 0.0021862, 0.0220866, 0.0017432, 0.0216876, -0.0195013, 0.0203434
5: -0.0038168, 0.0267420, -0.0037572, 0.0248777, -0.0286945, 0.0304993
6: 0.9918048, 1.0154709, 0.9916836, 1.0142148, -0.0224099, 0.0237873
7: -0.0094253, 0.0265976, -0.0102273, 0.0258754, -0.0332762, 0.0341488
8: -0.0032501, 0.0093212, -0.0026849, 0.0090949, -0.0123450, 0.0120061
9: -0.0391176, -0.0034082, -0.0378360, -0.0029067, -0.0362110, 0.0344278

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=74, inp2_unstable=69, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 193

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0190978, upper bound: 0.0192897
time: 2.85 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0190978, upper bound: 0.0208769
time: 2.88 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0018649, 0.0091400, -0.0014014, 0.0090974, -0.0109622, 0.0105415
1: -0.0056115, 0.0057855, -0.0051700, 0.0057029, -0.0113144, 0.0109555
2: -0.0392640, 0.0154110, -0.0381591, 0.0150563, -0.0543203, 0.0535701
3: -0.0038890, 0.0196964, -0.0038573, 0.0187376, -0.0226266, 0.0235537
4: 0.0024675, 0.0220321, 0.0026212, 0.0217431, -0.0192757, 0.0194109
5: -0.0038087, 0.0264875, -0.0037655, 0.0251372, -0.0289459, 0.0302530
6: 0.9918818, 1.0152993, 0.9919240, 1.0143896, -0.0225078, 0.0233753
7: -0.0089163, 0.0264990, -0.0086381, 0.0259759, -0.0326500, 0.0324686
8: -0.0031729, 0.0092903, -0.0027635, 0.0091264, -0.0122993, 0.0120538
9: -0.0389426, -0.0037265, -0.0380144, -0.0039004, -0.0350422, 0.0342879

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=69, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 193

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0198491, upper bound: 0.0198457
time: 2.78 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0198491, upper bound: 0.0211486
time: 3.56 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0017306, 0.0091277, -0.0012968, 0.0090877, -0.0108184, 0.0104245
1: -0.0054836, 0.0057616, -0.0050704, 0.0056842, -0.0111678, 0.0108320
2: -0.0389440, 0.0150847, -0.0379098, 0.0166918, -0.0556358, 0.0529945
3: -0.0038599, 0.0194187, -0.0040034, 0.0185213, -0.0223811, 0.0234221
4: 0.0026088, 0.0219484, 0.0019124, 0.0216779, -0.0190691, 0.0200360
5: -0.0037962, 0.0260963, -0.0037558, 0.0248326, -0.0286288, 0.0298521
6: 0.9919206, 1.0150357, 0.9917299, 1.0141844, -0.0222638, 0.0233059
7: -0.0086604, 0.0263475, -0.0099210, 0.0258579, -0.0325289, 0.0335264
8: -0.0030543, 0.0092428, -0.0026713, 0.0090895, -0.0121438, 0.0119141
9: -0.0386737, -0.0038865, -0.0378050, -0.0030982, -0.0355755, 0.0339185

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=69, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 193

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0197248, upper bound: 0.0197956
time: 3.43 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0197248, upper bound: 0.0211005
time: 3.32 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0025020, 0.0091987, -0.0014014, 0.0090974, -0.0115994, 0.0106001
1: -0.0062185, 0.0058992, -0.0051700, 0.0057029, -0.0119214, 0.0110692
2: -0.0407832, 0.0163262, -0.0381591, 0.0150563, -0.0558394, 0.0544853
3: -0.0039707, 0.0210147, -0.0038573, 0.0187376, -0.0227083, 0.0248720
4: 0.0020709, 0.0224293, 0.0026212, 0.0217431, -0.0196723, 0.0198082
5: -0.0038680, 0.0283438, -0.0037655, 0.0251372, -0.0290052, 0.0321093
6: 0.9917733, 1.0165501, 0.9919240, 1.0143896, -0.0226163, 0.0246261
7: -0.0096342, 0.0272181, -0.0086381, 0.0259759, -0.0334815, 0.0334291
8: -0.0037357, 0.0095156, -0.0027635, 0.0091264, -0.0128622, 0.0122791
9: -0.0402188, -0.0032775, -0.0380144, -0.0039004, -0.0363183, 0.0347368

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=69, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 132

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0191766, upper bound: 0.0190844
time: 2.58 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0191766, upper bound: 0.0209238
time: 2.84 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0023817, 0.0091876, -0.0012968, 0.0090877, -0.0114694, 0.0104845
1: -0.0061038, 0.0058777, -0.0050704, 0.0056842, -0.0117880, 0.0109481
2: -0.0404962, 0.0160018, -0.0379098, 0.0166918, -0.0571880, 0.0539116
3: -0.0039418, 0.0207657, -0.0040034, 0.0185213, -0.0224630, 0.0247691
4: 0.0022114, 0.0223543, 0.0019124, 0.0216779, -0.0194665, 0.0204419
5: -0.0038568, 0.0279932, -0.0037558, 0.0248326, -0.0286894, 0.0317490
6: 0.9918118, 1.0163138, 0.9917299, 1.0141844, -0.0223726, 0.0245839
7: -0.0093798, 0.0270823, -0.0099210, 0.0258579, -0.0333695, 0.0345080
8: -0.0036293, 0.0094730, -0.0026713, 0.0090895, -0.0127188, 0.0121443
9: -0.0399777, -0.0034366, -0.0378050, -0.0030982, -0.0368795, 0.0343683

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=74, inp2_unstable=69, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 132

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0190335, upper bound: 0.0190261
time: 2.97 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0190335, upper bound: 0.0208699
time: 2.57 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0014516, 0.0091020, -0.0020334, 0.0091556, -0.0106072, 0.0111354
1: -0.0052178, 0.0057118, -0.0057721, 0.0058156, -0.0110334, 0.0114839
2: -0.0382787, 0.0148868, -0.0396659, 0.0164693, -0.0547480, 0.0545527
3: -0.0038422, 0.0188414, -0.0039835, 0.0200451, -0.0238873, 0.0228249
4: 0.0026946, 0.0217744, 0.0020088, 0.0221372, -0.0194426, 0.0197656
5: -0.0037702, 0.0252834, -0.0038244, 0.0269785, -0.0307487, 0.0291077
6: 0.9919441, 1.0144880, 0.9917564, 1.0156301, -0.0236861, 0.0227316
7: -0.0085052, 0.0260326, -0.0097465, 0.0266892, -0.0329977, 0.0340604
8: -0.0028079, 0.0091442, -0.0033217, 0.0093499, -0.0121577, 0.0124659
9: -0.0381149, -0.0039835, -0.0392802, -0.0032073, -0.0349075, 0.0352967

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=69, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 188

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 78

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0192332, upper bound: 0.0206190
time: 5.82 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0192331, upper bound: 0.0206889
time: 4.01 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0013102, 0.0090890, -0.0019703, 0.0091498, -0.0104600, 0.0110593
1: -0.0050832, 0.0056866, -0.0057120, 0.0058043, -0.0108875, 0.0113986
2: -0.0379417, 0.0145627, -0.0395155, 0.0181607, -0.0561024, 0.0540781
3: -0.0038132, 0.0185490, -0.0041346, 0.0199146, -0.0237278, 0.0226836
4: 0.0028350, 0.0216863, 0.0012759, 0.0220978, -0.0192628, 0.0204104
5: -0.0037570, 0.0248716, -0.0038185, 0.0267947, -0.0305517, 0.0286901
6: 0.9919825, 1.0142106, 0.9915556, 1.0155063, -0.0235237, 0.0226550
7: -0.0082509, 0.0258731, -0.0110733, 0.0266180, -0.0328844, 0.0350612
8: -0.0026831, 0.0090942, -0.0032660, 0.0093276, -0.0120107, 0.0123602
9: -0.0378318, -0.0041425, -0.0391538, -0.0023777, -0.0354541, 0.0350113

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=68, inp2_unstable=76, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 193

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 78

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0187210, upper bound: 0.0205398
time: 3.15 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0187094, upper bound: 0.0206103
time: 3.08 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0018467, 0.0091384, -0.0020185, 0.0091542, -0.0110009, 0.0111569
1: -0.0055943, 0.0057823, -0.0057579, 0.0058129, -0.0114072, 0.0115402
2: -0.0392209, 0.0148706, -0.0396303, 0.0161664, -0.0553873, 0.0545010
3: -0.0038407, 0.0196590, -0.0039565, 0.0200143, -0.0238550, 0.0236154
4: 0.0027016, 0.0220208, 0.0021401, 0.0221279, -0.0194263, 0.0198807
5: -0.0038070, 0.0264347, -0.0038230, 0.0269351, -0.0307421, 0.0302577
6: 0.9919460, 1.0152637, 0.9917923, 1.0156009, -0.0236549, 0.0234714
7: -0.0084925, 0.0264786, -0.0095089, 0.0266724, -0.0331166, 0.0344580
8: -0.0031569, 0.0092839, -0.0033086, 0.0093446, -0.0125016, 0.0125925
9: -0.0389064, -0.0039915, -0.0392503, -0.0033559, -0.0355505, 0.0352589

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=74, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 193

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 78

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0192090, upper bound: 0.0205742
time: 2.89 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0192090, upper bound: 0.0206812
time: 3.45 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0017126, 0.0091260, -0.0019554, 0.0091484, -0.0108609, 0.0110815
1: -0.0054664, 0.0057584, -0.0056978, 0.0058017, -0.0112681, 0.0114562
2: -0.0389009, 0.0145403, -0.0394800, 0.0178724, -0.0567733, 0.0540203
3: -0.0038112, 0.0193813, -0.0041088, 0.0198839, -0.0236951, 0.0234902
4: 0.0028448, 0.0219371, 0.0014008, 0.0220886, -0.0192438, 0.0205363
5: -0.0037945, 0.0260437, -0.0038171, 0.0267514, -0.0305459, 0.0298608
6: 0.9919851, 1.0150005, 0.9915899, 1.0154772, -0.0234920, 0.0234106
7: -0.0082333, 0.0263271, -0.0108471, 0.0266012, -0.0329971, 0.0354798
8: -0.0030384, 0.0092364, -0.0032530, 0.0093223, -0.0123607, 0.0124894
9: -0.0386376, -0.0041535, -0.0391241, -0.0025192, -0.0361184, 0.0349706

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=75, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 188

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 78

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0187075, upper bound: 0.0205060
time: 2.70 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0187039, upper bound: 0.0206061
time: 3.09 seconds

## Summary of splitting at layer (split count: 10)
- Time for IS candidates: 7.22 seconds
IS_A1_B1_A1_B1_A2_B1_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 11, time: 7.22
Output dim: 6, lower bound: -0.0199117, upper bound: 0.0199992
IS_A1_B1_A1_B1_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 11, time: 7.22
Output dim: 6, lower bound: -0.0199117, upper bound: 0.0211564
IS_A1_B1_A1_B1_A2_B1_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 11, time: 7.22
Output dim: 6, lower bound: -0.0197716, upper bound: 0.0199346
IS_A1_B1_A1_B1_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 11, time: 7.22
Output dim: 6, lower bound: -0.0197716, upper bound: 0.0211061
IS_A1_B1_A1_B1_A2_B1_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 11, time: 7.22
Output dim: 6, lower bound: -0.0192940, upper bound: 0.0193615
IS_A1_B1_A1_B1_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 7.22
Output dim: 6, lower bound: -0.0192940, upper bound: 0.0209349
IS_A1_B1_A1_B1_A2_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 11, time: 7.22
Output dim: 6, lower bound: -0.0190978, upper bound: 0.0192897
IS_A1_B1_A1_B1_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 7.22
Output dim: 6, lower bound: -0.0190978, upper bound: 0.0208769
IS_A1_B1_A1_B1_A2_B1_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 11, time: 7.22
Output dim: 6, lower bound: -0.0198491, upper bound: 0.0198457
IS_A1_B1_A1_B1_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 11, time: 7.22
Output dim: 6, lower bound: -0.0198491, upper bound: 0.0211486
IS_A1_B1_A1_B1_A2_B1_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 11, time: 7.22
Output dim: 6, lower bound: -0.0197248, upper bound: 0.0197956
IS_A1_B1_A1_B1_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 11, time: 7.22
Output dim: 6, lower bound: -0.0197248, upper bound: 0.0211005
IS_A1_B1_A1_B1_A2_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 11, time: 7.22
Output dim: 6, lower bound: -0.0191766, upper bound: 0.0190844
IS_A1_B1_A1_B1_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 7.22
Output dim: 6, lower bound: -0.0191766, upper bound: 0.0209238
IS_A1_B1_A1_B1_A2_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 11, time: 7.22
Output dim: 6, lower bound: -0.0190335, upper bound: 0.0190261
IS_A1_B1_A1_B1_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 7.22
Output dim: 6, lower bound: -0.0190335, upper bound: 0.0208699
IS_A1_B1_A1_B1_A2_B2_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 11, time: 7.22
Output dim: 6, lower bound: -0.0192332, upper bound: 0.0206190
IS_A1_B1_A1_B1_A2_B2_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 11, time: 7.22
Output dim: 6, lower bound: -0.0192331, upper bound: 0.0206889
IS_A1_B1_A1_B1_A2_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 11, time: 7.22
Output dim: 6, lower bound: -0.0187210, upper bound: 0.0205398
IS_A1_B1_A1_B1_A2_B2_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 11, time: 7.22
Output dim: 6, lower bound: -0.0187094, upper bound: 0.0206103
IS_A1_B1_A1_B1_A2_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 11, time: 7.22
Output dim: 6, lower bound: -0.0192090, upper bound: 0.0205742
IS_A1_B1_A1_B1_A2_B2_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 11, time: 7.22
Output dim: 6, lower bound: -0.0192090, upper bound: 0.0206812
IS_A1_B1_A1_B1_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 11, time: 7.22
Output dim: 6, lower bound: -0.0187075, upper bound: 0.0205060
IS_A1_B1_A1_B1_A2_B2_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 11, time: 7.22
Output dim: 6, lower bound: -0.0187039, upper bound: 0.0206061

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0014388, 0.0091008, -0.0014167, 0.0090988, -0.0105375, 0.0105175
1: -0.0052056, 0.0057095, -0.0051846, 0.0057056, -0.0109112, 0.0108941
2: -0.0382482, 0.0148987, -0.0381956, 0.0153414, -0.0535896, 0.0530944
3: -0.0038432, 0.0188149, -0.0038828, 0.0187693, -0.0226125, 0.0226977
4: 0.0026894, 0.0217664, 0.0024976, 0.0217527, -0.0190633, 0.0192688
5: -0.0037690, 0.0252461, -0.0037670, 0.0251818, -0.0289508, 0.0290131
6: 0.9919426, 1.0144631, 0.9918901, 1.0144196, -0.0224769, 0.0225729
7: -0.0085145, 0.0260181, -0.0088617, 0.0259932, -0.0321707, 0.0320092
8: -0.0027966, 0.0091396, -0.0027771, 0.0091318, -0.0119284, 0.0119167
9: -0.0380893, -0.0039777, -0.0380451, -0.0037606, -0.0343287, 0.0340674

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=33, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=68, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 187

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 78

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0196234, upper bound: 0.0209205
time: 3.59 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0196779, upper bound: 0.0209387
time: 8.72 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0012979, 0.0090878, -0.0013123, 0.0090892, -0.0103870, 0.0104002
1: -0.0050714, 0.0056844, -0.0050851, 0.0056870, -0.0107583, 0.0107695
2: -0.0379122, 0.0145804, -0.0379467, 0.0170823, -0.0549944, 0.0525271
3: -0.0038148, 0.0185233, -0.0040383, 0.0185533, -0.0223681, 0.0225616
4: 0.0028274, 0.0216786, 0.0017432, 0.0216876, -0.0188602, 0.0199353
5: -0.0037559, 0.0248355, -0.0037572, 0.0248777, -0.0286336, 0.0285927
6: 0.9919804, 1.0141863, 0.9916836, 1.0142148, -0.0222344, 0.0225027
7: -0.0082648, 0.0258591, -0.0102273, 0.0258754, -0.0320514, 0.0331488
8: -0.0026721, 0.0090898, -0.0026849, 0.0090949, -0.0117671, 0.0117747
9: -0.0378070, -0.0041338, -0.0378360, -0.0029067, -0.0349003, 0.0337021

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=33, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=68, inp2_unstable=69, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 193

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0196213, upper bound: 0.0209444
time: 3.35 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0196264, upper bound: 0.0209917
time: 3.45 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0020496, 0.0091571, -0.0014167, 0.0090988, -0.0111484, 0.0105738
1: -0.0057876, 0.0058185, -0.0051846, 0.0057056, -0.0114932, 0.0110031
2: -0.0397047, 0.0158466, -0.0381956, 0.0153414, -0.0550460, 0.0540423
3: -0.0039279, 0.0200788, -0.0038828, 0.0187693, -0.0226972, 0.0239615
4: 0.0022787, 0.0221473, 0.0024976, 0.0217527, -0.0194740, 0.0196497
5: -0.0038259, 0.0270259, -0.0037670, 0.0251818, -0.0290077, 0.0307928
6: 0.9918302, 1.0156621, 0.9918901, 1.0144196, -0.0225893, 0.0237719
7: -0.0092581, 0.0267076, -0.0088617, 0.0259932, -0.0330197, 0.0329469
8: -0.0033362, 0.0093556, -0.0027771, 0.0091318, -0.0124680, 0.0121327
9: -0.0393128, -0.0035128, -0.0380451, -0.0037606, -0.0355522, 0.0345323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=33, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=74, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 187

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 78

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0189972, upper bound: 0.0206975
time: 3.82 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0190418, upper bound: 0.0207181
time: 2.68 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0019212, 0.0091452, -0.0013123, 0.0090892, -0.0110104, 0.0104576
1: -0.0056652, 0.0057956, -0.0050851, 0.0056870, -0.0113521, 0.0108807
2: -0.0393983, 0.0155263, -0.0379467, 0.0170823, -0.0564806, 0.0534730
3: -0.0038993, 0.0198130, -0.0040383, 0.0185533, -0.0224526, 0.0238512
4: 0.0024175, 0.0220672, 0.0017432, 0.0216876, -0.0192701, 0.0203240
5: -0.0038139, 0.0266516, -0.0037572, 0.0248777, -0.0286916, 0.0304088
6: 0.9918681, 1.0154098, 0.9916836, 1.0142148, -0.0223466, 0.0237262
7: -0.0090068, 0.0265626, -0.0102273, 0.0258754, -0.0329010, 0.0341130
8: -0.0032227, 0.0093102, -0.0026849, 0.0090949, -0.0123176, 0.0119951
9: -0.0390554, -0.0036699, -0.0378360, -0.0029067, -0.0361487, 0.0341661

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=33, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=73, inp2_unstable=69, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 193

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0189124, upper bound: 0.0207165
time: 2.59 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0189104, upper bound: 0.0207584
time: 2.68 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0018326, 0.0091371, -0.0014014, 0.0090974, -0.0109299, 0.0105385
1: -0.0055808, 0.0057798, -0.0051700, 0.0057029, -0.0112836, 0.0109498
2: -0.0391871, 0.0148828, -0.0381591, 0.0150563, -0.0542434, 0.0530419
3: -0.0038418, 0.0196296, -0.0038573, 0.0187376, -0.0225794, 0.0234870
4: 0.0026963, 0.0220120, 0.0026212, 0.0217431, -0.0190468, 0.0193908
5: -0.0038057, 0.0263934, -0.0037655, 0.0251372, -0.0289429, 0.0301589
6: 0.9919445, 1.0152360, 0.9919240, 1.0143896, -0.0224451, 0.0233120
7: -0.0085021, 0.0264626, -0.0086381, 0.0259759, -0.0322742, 0.0324321
8: -0.0031444, 0.0092789, -0.0027635, 0.0091264, -0.0122708, 0.0120424
9: -0.0388780, -0.0039855, -0.0380144, -0.0039004, -0.0349776, 0.0340289

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=33, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=69, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 187

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 78

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0195783, upper bound: 0.0209199
time: 3.11 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0196078, upper bound: 0.0209385
time: 3.07 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0016985, 0.0091247, -0.0012968, 0.0090877, -0.0107862, 0.0104216
1: -0.0054530, 0.0057559, -0.0050704, 0.0056842, -0.0111373, 0.0108263
2: -0.0388674, 0.0145573, -0.0379098, 0.0166918, -0.0555593, 0.0524671
3: -0.0038128, 0.0193523, -0.0040034, 0.0185213, -0.0223340, 0.0233557
4: 0.0028374, 0.0219284, 0.0019124, 0.0216779, -0.0188405, 0.0200160
5: -0.0037932, 0.0260028, -0.0037558, 0.0248326, -0.0286258, 0.0297586
6: 0.9919831, 1.0149729, 0.9917299, 1.0141844, -0.0222012, 0.0232431
7: -0.0082467, 0.0263112, -0.0099210, 0.0258579, -0.0321486, 0.0334904
8: -0.0030260, 0.0092315, -0.0026713, 0.0090895, -0.0121155, 0.0119027
9: -0.0386095, -0.0041452, -0.0378050, -0.0030982, -0.0355112, 0.0336598

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=33, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=69, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 193

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0195682, upper bound: 0.0209392
time: 3.25 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0195791, upper bound: 0.0209875
time: 2.99 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0024711, 0.0091959, -0.0014014, 0.0090974, -0.0115684, 0.0105973
1: -0.0061890, 0.0058937, -0.0051700, 0.0057029, -0.0118919, 0.0110637
2: -0.0407093, 0.0157930, -0.0381591, 0.0150563, -0.0557656, 0.0539521
3: -0.0039231, 0.0209506, -0.0038573, 0.0187376, -0.0226607, 0.0248079
4: 0.0023019, 0.0224100, 0.0026212, 0.0217431, -0.0194412, 0.0197889
5: -0.0038651, 0.0282536, -0.0037655, 0.0251372, -0.0290023, 0.0320191
6: 0.9918365, 1.0164893, 0.9919240, 1.0143896, -0.0225531, 0.0245653
7: -0.0092160, 0.0271832, -0.0086381, 0.0259759, -0.0331027, 0.0333938
8: -0.0037083, 0.0095046, -0.0027635, 0.0091264, -0.0128348, 0.0122682
9: -0.0401568, -0.0035391, -0.0380144, -0.0039004, -0.0362564, 0.0344753

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=33, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=75, inp2_unstable=69, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 187

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 78

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0189120, upper bound: 0.0206938
time: 3.06 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0189239, upper bound: 0.0207113
time: 2.96 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0023506, 0.0091848, -0.0012968, 0.0090877, -0.0114383, 0.0104816
1: -0.0060743, 0.0058722, -0.0050704, 0.0056842, -0.0117585, 0.0109426
2: -0.0404222, 0.0154703, -0.0379098, 0.0166918, -0.0571140, 0.0533802
3: -0.0038943, 0.0207014, -0.0040034, 0.0185213, -0.0224156, 0.0247048
4: 0.0024417, 0.0223349, 0.0019124, 0.0216779, -0.0192362, 0.0204225
5: -0.0038539, 0.0279027, -0.0037558, 0.0248326, -0.0286865, 0.0316585
6: 0.9918748, 1.0162528, 0.9917299, 1.0141844, -0.0223095, 0.0245229
7: -0.0089629, 0.0270472, -0.0099210, 0.0258579, -0.0329838, 0.0344726
8: -0.0036020, 0.0094621, -0.0026713, 0.0090895, -0.0126914, 0.0121333
9: -0.0399156, -0.0036973, -0.0378050, -0.0030982, -0.0368174, 0.0341076

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=33, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=74, inp2_unstable=69, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 193

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0188055, upper bound: 0.0207073
time: 3.01 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0188073, upper bound: 0.0207508
time: 2.73 seconds

## Summary of splitting at layer (split count: 11)
- Time for IS candidates: 7.15 seconds
IS_A1_B1_A1_B1_A2_B1_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 12, time: 7.15
Output dim: 6, lower bound: -0.0196234, upper bound: 0.0209205
IS_A1_B1_A1_B1_A2_B1_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 12, time: 7.15
Output dim: 6, lower bound: -0.0196779, upper bound: 0.0209387
IS_A1_B1_A1_B1_A2_B1_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 12, time: 7.15
Output dim: 6, lower bound: -0.0196213, upper bound: 0.0209444
IS_A1_B1_A1_B1_A2_B1_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 12, time: 7.15
Output dim: 6, lower bound: -0.0196264, upper bound: 0.0209917
IS_A1_B1_A1_B1_A2_B1_A1_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 12, time: 7.15
Output dim: 6, lower bound: -0.0189972, upper bound: 0.0206975
IS_A1_B1_A1_B1_A2_B1_A1_B2_A2_B1_A2_B2, status: Status.VERIFIED, split count: 12, time: 7.15
Output dim: 6, lower bound: -0.0190418, upper bound: 0.0207181
IS_A1_B1_A1_B1_A2_B1_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 12, time: 7.15
Output dim: 6, lower bound: -0.0189124, upper bound: 0.0207165
IS_A1_B1_A1_B1_A2_B1_A1_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 12, time: 7.15
Output dim: 6, lower bound: -0.0189104, upper bound: 0.0207584
IS_A1_B1_A1_B1_A2_B1_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 12, time: 7.15
Output dim: 6, lower bound: -0.0195783, upper bound: 0.0209199
IS_A1_B1_A1_B1_A2_B1_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 12, time: 7.15
Output dim: 6, lower bound: -0.0196078, upper bound: 0.0209385
IS_A1_B1_A1_B1_A2_B1_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 12, time: 7.15
Output dim: 6, lower bound: -0.0195682, upper bound: 0.0209392
IS_A1_B1_A1_B1_A2_B1_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 12, time: 7.15
Output dim: 6, lower bound: -0.0195791, upper bound: 0.0209875
IS_A1_B1_A1_B1_A2_B1_A2_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 12, time: 7.15
Output dim: 6, lower bound: -0.0189120, upper bound: 0.0206938
IS_A1_B1_A1_B1_A2_B1_A2_B2_A2_B1_A2_B2, status: Status.VERIFIED, split count: 12, time: 7.15
Output dim: 6, lower bound: -0.0189239, upper bound: 0.0207113
IS_A1_B1_A1_B1_A2_B1_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 12, time: 7.15
Output dim: 6, lower bound: -0.0188055, upper bound: 0.0207073
IS_A1_B1_A1_B1_A2_B1_A2_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 12, time: 7.15
Output dim: 6, lower bound: -0.0188073, upper bound: 0.0207508

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0013125, 0.0090892, -0.0010697, 0.0090668, -0.0103793, 0.0101589
1: -0.0050853, 0.0056870, -0.0048540, 0.0056437, -0.0107290, 0.0105410
2: -0.0379472, 0.0146120, -0.0373683, 0.0145293, -0.0524765, 0.0519803
3: -0.0038176, 0.0185537, -0.0038103, 0.0180514, -0.0218690, 0.0223640
4: 0.0028137, 0.0216877, 0.0028495, 0.0215363, -0.0187226, 0.0188382
5: -0.0037573, 0.0248783, -0.0037347, 0.0241709, -0.0279281, 0.0286129
6: 0.9919767, 1.0142150, 0.9919865, 1.0137385, -0.0217618, 0.0222285
7: -0.0082896, 0.0258756, -0.0082247, 0.0256016, -0.0314947, 0.0311372
8: -0.0026851, 0.0090950, -0.0024706, 0.0090091, -0.0116942, 0.0115656
9: -0.0378364, -0.0041183, -0.0373501, -0.0041589, -0.0336775, 0.0332317

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=69, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 193

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B2_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0203182, upper bound: 0.0204599
time: 3.61 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B2_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0202453, upper bound: 0.0203473
time: 3.02 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0012666, 0.0090849, -0.0010735, 0.0090672, -0.0103338, 0.0101585
1: -0.0050416, 0.0056788, -0.0048577, 0.0056444, -0.0106860, 0.0105365
2: -0.0378377, 0.0144892, -0.0373774, 0.0152211, -0.0530588, 0.0518666
3: -0.0038067, 0.0184587, -0.0038720, 0.0180593, -0.0218659, 0.0223307
4: 0.0028669, 0.0216591, 0.0025497, 0.0215387, -0.0186718, 0.0191094
5: -0.0037530, 0.0247444, -0.0037350, 0.0241820, -0.0279350, 0.0284795
6: 0.9919913, 1.0141250, 0.9919043, 1.0137459, -0.0217546, 0.0222207
7: -0.0081933, 0.0258238, -0.0087674, 0.0256059, -0.0315195, 0.0314746
8: -0.0026445, 0.0090788, -0.0024740, 0.0090105, -0.0116550, 0.0115528
9: -0.0377444, -0.0041786, -0.0373577, -0.0038196, -0.0339248, 0.0331791

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=67, inp2_unstable=71, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 193

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0203817, upper bound: 0.0204645
time: 4.21 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B2_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0203142, upper bound: 0.0203474
time: 4.51 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0011660, 0.0090757, -0.0009483, 0.0090556, -0.0102216, 0.0100240
1: -0.0049457, 0.0056609, -0.0047384, 0.0056220, -0.0105678, 0.0103992
2: -0.0375978, 0.0143658, -0.0370788, 0.0164886, -0.0540864, 0.0514446
3: -0.0037956, 0.0182505, -0.0039852, 0.0178001, -0.0215958, 0.0222358
4: 0.0029204, 0.0215963, 0.0020005, 0.0214606, -0.0185402, 0.0195959
5: -0.0037436, 0.0244513, -0.0037234, 0.0238171, -0.0275607, 0.0281747
6: 0.9920059, 1.0139275, 0.9917541, 1.0135001, -0.0214942, 0.0221734
7: -0.0080964, 0.0257102, -0.0097616, 0.0254646, -0.0315090, 0.0325676
8: -0.0025556, 0.0090432, -0.0023634, 0.0089662, -0.0115219, 0.0114065
9: -0.0375428, -0.0042391, -0.0371069, -0.0031979, -0.0343450, 0.0328678

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=68, inp2_unstable=68, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 193

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 78

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0206107, upper bound: 0.0206539
time: 4.00 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B2_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0206194, upper bound: 0.0207244
time: 3.47 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0011761, 0.0090766, -0.0011090, 0.0090704, -0.0102466, 0.0101856
1: -0.0049554, 0.0056627, -0.0048915, 0.0056507, -0.0106061, 0.0105541
2: -0.0376220, 0.0143950, -0.0374620, 0.0175658, -0.0551877, 0.0518570
3: -0.0037983, 0.0182715, -0.0040815, 0.0181327, -0.0219309, 0.0223529
4: 0.0029077, 0.0216027, 0.0015337, 0.0215608, -0.0186531, 0.0200690
5: -0.0037446, 0.0244808, -0.0037383, 0.0242854, -0.0280299, 0.0282191
6: 0.9920024, 1.0139472, 0.9916262, 1.0138156, -0.0218132, 0.0223210
7: -0.0081194, 0.0257217, -0.0106066, 0.0256459, -0.0316686, 0.0334151
8: -0.0025646, 0.0090468, -0.0025053, 0.0090230, -0.0115876, 0.0115521
9: -0.0375631, -0.0042248, -0.0374288, -0.0026695, -0.0348936, 0.0332040

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=68, inp2_unstable=71, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 193

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 78

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0207792, upper bound: 0.0206960
time: 3.48 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0207887, upper bound: 0.0207753
time: 4.26 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0017117, 0.0091259, -0.0010538, 0.0090654, -0.0107770, 0.0101797
1: -0.0054656, 0.0057582, -0.0048389, 0.0056409, -0.0111065, 0.0105971
2: -0.0388989, 0.0145893, -0.0373304, 0.0142370, -0.0531358, 0.0519197
3: -0.0038156, 0.0193796, -0.0037841, 0.0180185, -0.0218341, 0.0231637
4: 0.0028235, 0.0219366, 0.0029762, 0.0215264, -0.0187029, 0.0189604
5: -0.0037944, 0.0260413, -0.0037332, 0.0241246, -0.0279190, 0.0297744
6: 0.9919794, 1.0149987, 0.9920211, 1.0137072, -0.0217278, 0.0229775
7: -0.0082718, 0.0263261, -0.0079954, 0.0255837, -0.0315933, 0.0315594
8: -0.0030376, 0.0092361, -0.0024566, 0.0090035, -0.0120412, 0.0116927
9: -0.0386359, -0.0041295, -0.0373182, -0.0043023, -0.0343336, 0.0331888

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=68, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 193

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B2_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0203049, upper bound: 0.0204529
time: 3.46 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B2_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0202277, upper bound: 0.0203127
time: 3.02 seconds

## Summary of splitting at layer (split count: 12)
- Time for IS candidates: 7.87 seconds
IS_A1_B1_A1_B1_A2_B1_A1_B2_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 13, time: 7.87
Output dim: 6, lower bound: -0.0203182, upper bound: 0.0204599
IS_A1_B1_A1_B1_A2_B1_A1_B2_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 13, time: 7.87
Output dim: 6, lower bound: -0.0202453, upper bound: 0.0203473
IS_A1_B1_A1_B1_A2_B1_A1_B2_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 13, time: 7.87
Output dim: 6, lower bound: -0.0203817, upper bound: 0.0204645
IS_A1_B1_A1_B1_A2_B1_A1_B2_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 13, time: 7.87
Output dim: 6, lower bound: -0.0203142, upper bound: 0.0203474
IS_A1_B1_A1_B1_A2_B1_A1_B2_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 13, time: 7.87
Output dim: 6, lower bound: -0.0206107, upper bound: 0.0206539
IS_A1_B1_A1_B1_A2_B1_A1_B2_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 13, time: 7.87
Output dim: 6, lower bound: -0.0206194, upper bound: 0.0207244
IS_A1_B1_A1_B1_A2_B1_A1_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 13, time: 7.87
Output dim: 6, lower bound: -0.0207792, upper bound: 0.0206960
IS_A1_B1_A1_B1_A2_B1_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 13, time: 7.87
Output dim: 6, lower bound: -0.0207887, upper bound: 0.0207753
IS_A1_B1_A1_B1_A2_B1_A2_B2_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 13, time: 7.87
Output dim: 6, lower bound: -0.0203049, upper bound: 0.0204529
IS_A1_B1_A1_B1_A2_B1_A2_B2_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 13, time: 7.87
Output dim: 6, lower bound: -0.0202277, upper bound: 0.0203127
IS_A1_B1_A1_B1_A2_B1_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 12, time: 7.87
Output dim: 6, lower bound: -0.0196078, upper bound: 0.0209385
IS_A1_B1_A1_B1_A2_B1_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 12, time: 7.87
Output dim: 6, lower bound: -0.0195682, upper bound: 0.0209392
IS_A1_B1_A1_B1_A2_B1_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 12, time: 7.87
Output dim: 6, lower bound: -0.0195791, upper bound: 0.0209875

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 6.52 + 593.60 = 600.12 seconds
