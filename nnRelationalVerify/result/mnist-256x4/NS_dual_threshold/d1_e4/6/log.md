## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 6)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.017292288


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797)
1: (0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944)
2: (-0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0199057, 0.0199058)
3: (-0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347)
4: (-0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642)
5: (-0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152)
6: (-0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641)
7: (-0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542)
8: (-0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295680, 0.0295680)
9: (-0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.68 + 3.46 = 5.14 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.0180128, upper bound: 0.0180128

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 59

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0177198, upper bound: 0.0179069
time: 2.58 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0179069, upper bound: 0.0179069
time: 3.43 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 6.16 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 6.16
Output dim: 1, lower bound: -0.0177198, upper bound: 0.0179069
NS_A2, status: Status.UNKNOWN, split count: 1, time: 6.16
Output dim: 1, lower bound: -0.0179069, upper bound: 0.0179069

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.0085212, 0.0074265, -0.0089309, 0.0079629, -0.0164841, 0.0163574
1: 0.9906642, 1.0139350, 0.9903470, 1.0147487, -0.0240845, 0.0235879
2: -0.0109579, 0.0076869, -0.0115093, 0.0081113, -0.0190348, 0.0191962
3: -0.0006711, 0.0052161, -0.0007859, 0.0053476, -0.0060187, 0.0060019
4: -0.0093614, 0.0070775, -0.0099008, 0.0075134, -0.0168748, 0.0169783
5: -0.0031012, 0.0112743, -0.0032879, 0.0119492, -0.0150504, 0.0145622
6: -0.0128261, 0.0022574, -0.0136593, 0.0023233, -0.0151495, 0.0159166
7: -0.0103273, 0.0014276, -0.0105491, 0.0017248, -0.0120521, 0.0119767
8: -0.0153844, 0.0126490, -0.0157998, 0.0133735, -0.0286150, 0.0283080
9: -0.0079615, 0.0086882, -0.0087046, 0.0089319, -0.0168934, 0.0173928

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0175767, upper bound: 0.0174265
time: 2.95 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0176041, upper bound: 0.0177991
time: 3.00 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -0.0089106, 0.0079363, -0.0089481, 0.0079855, -0.0168960, 0.0168844
1: 0.9902947, 1.0147082, 0.9901879, 1.0147827, -0.0244880, 0.0245203
2: -0.0116006, 0.0080902, -0.0117861, 0.0081291, -0.0196003, 0.0197677
3: -0.0007801, 0.0053693, -0.0007907, 0.0054135, -0.0061937, 0.0061599
4: -0.0098740, 0.0075855, -0.0099234, 0.0077321, -0.0176061, 0.0175089
5: -0.0032786, 0.0119156, -0.0032957, 0.0119775, -0.0152561, 0.0152113
6: -0.0136179, 0.0023200, -0.0136942, 0.0023261, -0.0159440, 0.0160142
7: -0.0105858, 0.0017100, -0.0106604, 0.0017373, -0.0123231, 0.0123705
8: -0.0157792, 0.0134933, -0.0158173, 0.0137370, -0.0293731, 0.0291429
9: -0.0086677, 0.0089198, -0.0087358, 0.0089422, -0.0176099, 0.0176556

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0177684, upper bound: 0.0174265
time: 2.57 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0177991, upper bound: 0.0177991
time: 3.24 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 7.36 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 7.36
Output dim: 1, lower bound: -0.0175767, upper bound: 0.0174265
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 7.36
Output dim: 1, lower bound: -0.0176041, upper bound: 0.0177991
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 7.36
Output dim: 1, lower bound: -0.0177684, upper bound: 0.0174265
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 7.36
Output dim: 1, lower bound: -0.0177991, upper bound: 0.0177991

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -0.0084400, 0.0073202, -0.0094411, 0.0086308, -0.0170708, 0.0167613
1: 0.9910872, 1.0137736, 0.9915989, 1.0157616, -0.0246744, 0.0221747
2: -0.0102219, 0.0076029, -0.0095871, 0.0086397, -0.0188615, 0.0171900
3: -0.0006484, 0.0050406, -0.0009287, 0.0048284, -0.0054768, 0.0059692
4: -0.0092546, 0.0064958, -0.0105723, 0.0057924, -0.0150470, 0.0170681
5: -0.0030643, 0.0111407, -0.0035202, 0.0127893, -0.0158536, 0.0146609
6: -0.0126611, 0.0022443, -0.0146965, 0.0024054, -0.0150665, 0.0169408
7: -0.0100311, 0.0013687, -0.0096731, 0.0020949, -0.0121260, 0.0110418
8: -0.0153021, 0.0116819, -0.0163171, 0.0105126, -0.0256724, 0.0278595
9: -0.0078143, 0.0086399, -0.0096298, 0.0092354, -0.0170497, 0.0182698

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0173860, upper bound: 0.0173031
time: 3.24 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0174820, upper bound: 0.0173130
time: 2.74 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -0.0085212, 0.0074265, -0.0088899, 0.0079092, -0.0164304, 0.0163164
1: 0.9906642, 1.0139350, 0.9904547, 1.0146670, -0.0240029, 0.0234802
2: -0.0109579, 0.0076869, -0.0113222, 0.0080688, -0.0189932, 0.0189814
3: -0.0006711, 0.0052161, -0.0007744, 0.0053029, -0.0059740, 0.0059904
4: -0.0093614, 0.0070775, -0.0098468, 0.0073654, -0.0167269, 0.0169243
5: -0.0031012, 0.0112743, -0.0032692, 0.0118816, -0.0149828, 0.0145435
6: -0.0128261, 0.0022574, -0.0135758, 0.0023167, -0.0151428, 0.0158332
7: -0.0103273, 0.0014276, -0.0104738, 0.0016951, -0.0120223, 0.0119014
8: -0.0153844, 0.0126490, -0.0157582, 0.0131276, -0.0283300, 0.0282666
9: -0.0079615, 0.0086882, -0.0086302, 0.0089075, -0.0168690, 0.0173184

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0174176, upper bound: 0.0176894
time: 3.32 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0175099, upper bound: 0.0176975
time: 2.62 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -0.0088305, 0.0078314, -0.0094594, 0.0086548, -0.0174852, 0.0172908
1: 0.9907191, 1.0145491, 0.9914380, 1.0157979, -0.0250788, 0.0231111
2: -0.0108622, 0.0080073, -0.0096116, 0.0086586, -0.0195208, 0.0176189
3: -0.0007577, 0.0051932, -0.0009338, 0.0048950, -0.0056528, 0.0061270
4: -0.0097686, 0.0070019, -0.0105963, 0.0060135, -0.0157821, 0.0175982
5: -0.0032421, 0.0117837, -0.0035286, 0.0128194, -0.0160616, 0.0153123
6: -0.0134550, 0.0023071, -0.0147337, 0.0024083, -0.0158634, 0.0170408
7: -0.0102888, 0.0016519, -0.0097856, 0.0021082, -0.0123969, 0.0114375
8: -0.0156980, 0.0125232, -0.0163356, 0.0108801, -0.0264364, 0.0286924
9: -0.0085225, 0.0088722, -0.0096630, 0.0092462, -0.0177687, 0.0185352

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0176008, upper bound: 0.0173031
time: 3.04 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0176671, upper bound: 0.0173130
time: 3.31 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -0.0089106, 0.0079363, -0.0089071, 0.0079317, -0.0168423, 0.0168434
1: 0.9902947, 1.0147082, 0.9902934, 1.0147012, -0.0244066, 0.0244148
2: -0.0116006, 0.0080902, -0.0116026, 0.0080866, -0.0195586, 0.0195323
3: -0.0007801, 0.0053693, -0.0007792, 0.0053698, -0.0061499, 0.0061485
4: -0.0098740, 0.0075855, -0.0098694, 0.0075870, -0.0174610, 0.0174549
5: -0.0032786, 0.0119156, -0.0032770, 0.0119099, -0.0151885, 0.0151926
6: -0.0136179, 0.0023200, -0.0136108, 0.0023195, -0.0159373, 0.0159308
7: -0.0105858, 0.0017100, -0.0105866, 0.0017075, -0.0122933, 0.0122966
8: -0.0157792, 0.0134933, -0.0157757, 0.0134960, -0.0290923, 0.0291014
9: -0.0086677, 0.0089198, -0.0086614, 0.0089178, -0.0175855, 0.0175812

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0176378, upper bound: 0.0176894
time: 3.09 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0176975, upper bound: 0.0176975
time: 2.82 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 8.10 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 8.10
Output dim: 1, lower bound: -0.0173860, upper bound: 0.0173031
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 8.10
Output dim: 1, lower bound: -0.0174820, upper bound: 0.0173130
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 8.10
Output dim: 1, lower bound: -0.0174176, upper bound: 0.0176894
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 8.10
Output dim: 1, lower bound: -0.0175099, upper bound: 0.0176975
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 8.10
Output dim: 1, lower bound: -0.0176008, upper bound: 0.0173031
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 8.10
Output dim: 1, lower bound: -0.0176671, upper bound: 0.0173130
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 8.10
Output dim: 1, lower bound: -0.0176378, upper bound: 0.0176894
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 8.10
Output dim: 1, lower bound: -0.0176975, upper bound: 0.0176975

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0085487, 0.0074625, -0.0093770, 0.0085470, -0.0170956, 0.0168396
1: 0.9914784, 1.0139896, 0.9917587, 1.0156343, -0.0241559, 0.0222309
2: -0.0095412, 0.0077154, -0.0095038, 0.0085733, -0.0181145, 0.0172192
3: -0.0006788, 0.0048783, -0.0009107, 0.0047620, -0.0054409, 0.0057890
4: -0.0093977, 0.0059579, -0.0104880, 0.0055726, -0.0149703, 0.0164458
5: -0.0031138, 0.0113197, -0.0034911, 0.0126838, -0.0157976, 0.0148107
6: -0.0128821, 0.0022618, -0.0145662, 0.0023951, -0.0152772, 0.0168281
7: -0.0097573, 0.0014476, -0.0095612, 0.0020484, -0.0118057, 0.0110088
8: -0.0154123, 0.0107876, -0.0162521, 0.0101472, -0.0254188, 0.0268985
9: -0.0080114, 0.0087046, -0.0095137, 0.0091973, -0.0172087, 0.0182182

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0171544, upper bound: 0.0170843
time: 2.73 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0171720, upper bound: 0.0170843
time: 3.04 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0083534, 0.0072069, -0.0094197, 0.0086029, -0.0169563, 0.0166266
1: 0.9912776, 1.0136019, 0.9916546, 1.0157192, -0.0244415, 0.0219473
2: -0.0098906, 0.0075132, -0.0095593, 0.0086175, -0.0185082, 0.0170725
3: -0.0006242, 0.0049616, -0.0009227, 0.0048052, -0.0054294, 0.0058842
4: -0.0091406, 0.0062340, -0.0105442, 0.0057157, -0.0148563, 0.0167782
5: -0.0030248, 0.0109980, -0.0035105, 0.0127542, -0.0157790, 0.0145085
6: -0.0124850, 0.0022304, -0.0146531, 0.0024020, -0.0148870, 0.0168835
7: -0.0098979, 0.0013059, -0.0096340, 0.0020794, -0.0119773, 0.0109399
8: -0.0152143, 0.0112467, -0.0162954, 0.0103850, -0.0254564, 0.0273986
9: -0.0076572, 0.0085884, -0.0095911, 0.0092227, -0.0168799, 0.0181795

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0172541, upper bound: 0.0170924
time: 2.82 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0172684, upper bound: 0.0170924
time: 3.27 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0086241, 0.0075613, -0.0088202, 0.0078180, -0.0164421, 0.0163815
1: 0.9910607, 1.0141393, 0.9906253, 1.0145288, -0.0234680, 0.0235140
2: -0.0102679, 0.0077935, -0.0110254, 0.0079966, -0.0182645, 0.0188190
3: -0.0006999, 0.0050515, -0.0007549, 0.0052322, -0.0059321, 0.0058064
4: -0.0094969, 0.0065322, -0.0097550, 0.0071309, -0.0166278, 0.0162872
5: -0.0031481, 0.0114438, -0.0032374, 0.0117668, -0.0149149, 0.0146813
6: -0.0130354, 0.0022739, -0.0134342, 0.0023055, -0.0153409, 0.0157081
7: -0.0100496, 0.0015023, -0.0103544, 0.0016445, -0.0116941, 0.0118567
8: -0.0154887, 0.0117424, -0.0156876, 0.0127377, -0.0280482, 0.0272878
9: -0.0081482, 0.0087494, -0.0085038, 0.0088661, -0.0170143, 0.0172533

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A1_B2_A1_A1

### Relational analysis result of NS_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0172037, upper bound: 0.0174620
time: 2.95 seconds

## Relational analysis of NS_A1_B2_A1_A2

### Relational analysis result of NS_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0172037, upper bound: 0.0174786
time: 3.22 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0084323, 0.0073101, -0.0088663, 0.0078784, -0.0163106, 0.0161765
1: 0.9908552, 1.0137584, 0.9905083, 1.0146204, -0.0237653, 0.0232502
2: -0.0106252, 0.0075949, -0.0112290, 0.0080444, -0.0186696, 0.0188170
3: -0.0006462, 0.0051367, -0.0007678, 0.0052807, -0.0059270, 0.0059045
4: -0.0092445, 0.0068146, -0.0098157, 0.0072918, -0.0165363, 0.0166303
5: -0.0030607, 0.0111280, -0.0032584, 0.0118428, -0.0149035, 0.0143864
6: -0.0126455, 0.0022431, -0.0135279, 0.0023129, -0.0149584, 0.0157710
7: -0.0101934, 0.0013631, -0.0104363, 0.0016780, -0.0118713, 0.0117994
8: -0.0152943, 0.0122118, -0.0157343, 0.0130052, -0.0281182, 0.0278008
9: -0.0078003, 0.0086354, -0.0085875, 0.0088935, -0.0166938, 0.0172228

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0172798, upper bound: 0.0174887
time: 2.51 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0172970, upper bound: 0.0174887
time: 2.75 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0089289, 0.0079602, -0.0093952, 0.0085707, -0.0174996, 0.0173554
1: 0.9911113, 1.0147445, 0.9915974, 1.0156704, -0.0245591, 0.0231472
2: -0.0101797, 0.0081092, -0.0095273, 0.0085921, -0.0187718, 0.0176365
3: -0.0007853, 0.0050305, -0.0009158, 0.0048289, -0.0056142, 0.0059463
4: -0.0098981, 0.0064625, -0.0105118, 0.0057943, -0.0156923, 0.0169743
5: -0.0032869, 0.0119458, -0.0034993, 0.0127137, -0.0160006, 0.0154451
6: -0.0136551, 0.0023230, -0.0146031, 0.0023980, -0.0160531, 0.0169261
7: -0.0100142, 0.0017233, -0.0096740, 0.0020616, -0.0120757, 0.0113973
8: -0.0157977, 0.0116265, -0.0162705, 0.0105156, -0.0261710, 0.0277329
9: -0.0087009, 0.0089307, -0.0095465, 0.0092080, -0.0179090, 0.0184773

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0173699, upper bound: 0.0170843
time: 2.81 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0173846, upper bound: 0.0170842
time: 3.31 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0087464, 0.0077213, -0.0094381, 0.0086269, -0.0173733, 0.0171595
1: 0.9909131, 1.0143818, 0.9914942, 1.0157557, -0.0248426, 0.0228876
2: -0.0105247, 0.0079202, -0.0095832, 0.0086366, -0.0191613, 0.0175034
3: -0.0007342, 0.0051128, -0.0009278, 0.0048717, -0.0056059, 0.0060406
4: -0.0096579, 0.0067352, -0.0105684, 0.0059362, -0.0155941, 0.0173035
5: -0.0032038, 0.0116452, -0.0035189, 0.0127844, -0.0159883, 0.0151641
6: -0.0132840, 0.0022936, -0.0146905, 0.0024049, -0.0156889, 0.0169841
7: -0.0101530, 0.0015909, -0.0097463, 0.0020927, -0.0122457, 0.0113372
8: -0.0156127, 0.0120798, -0.0163140, 0.0107516, -0.0262234, 0.0282226
9: -0.0083699, 0.0088222, -0.0096245, 0.0092336, -0.0176035, 0.0184466

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0174393, upper bound: 0.0170924
time: 2.86 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0174540, upper bound: 0.0170924
time: 3.09 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0090037, 0.0080583, -0.0088372, 0.0078402, -0.0168440, 0.0168955
1: 0.9906924, 1.0148932, 0.9904640, 1.0145625, -0.0238701, 0.0244291
2: -0.0109087, 0.0081867, -0.0113058, 0.0080142, -0.0189229, 0.0194108
3: -0.0008062, 0.0052043, -0.0007596, 0.0052990, -0.0061052, 0.0059639
4: -0.0099966, 0.0070386, -0.0097774, 0.0073525, -0.0173491, 0.0168161
5: -0.0033210, 0.0120690, -0.0032452, 0.0117948, -0.0151158, 0.0153142
6: -0.0138073, 0.0023350, -0.0134687, 0.0023082, -0.0161155, 0.0158037
7: -0.0103074, 0.0017776, -0.0104672, 0.0016568, -0.0119643, 0.0122448
8: -0.0158736, 0.0125843, -0.0157048, 0.0131061, -0.0287996, 0.0281236
9: -0.0088367, 0.0089752, -0.0085347, 0.0088762, -0.0177128, 0.0175099

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A2_B2_A1_A1

### Relational analysis result of NS_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0174220, upper bound: 0.0174620
time: 3.20 seconds

## Relational analysis of NS_A2_B2_A1_A2

### Relational analysis result of NS_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0174220, upper bound: 0.0174786
time: 2.59 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0088244, 0.0078235, -0.0088835, 0.0079009, -0.0167252, 0.0167070
1: 0.9904847, 1.0145369, 0.9903471, 1.0146544, -0.0241697, 0.0241897
2: -0.0112700, 0.0080010, -0.0115091, 0.0080622, -0.0192703, 0.0193688
3: -0.0007560, 0.0052905, -0.0007726, 0.0053475, -0.0061035, 0.0060630
4: -0.0097606, 0.0073242, -0.0098384, 0.0075132, -0.0172738, 0.0171626
5: -0.0032393, 0.0117737, -0.0032663, 0.0118710, -0.0151104, 0.0150400
6: -0.0134427, 0.0023062, -0.0135628, 0.0023157, -0.0157583, 0.0158690
7: -0.0104528, 0.0016475, -0.0105490, 0.0016904, -0.0121432, 0.0121966
8: -0.0156918, 0.0130590, -0.0157517, 0.0133732, -0.0288824, 0.0286369
9: -0.0085114, 0.0088686, -0.0086186, 0.0089037, -0.0174151, 0.0174872

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0174730, upper bound: 0.0174887
time: 3.07 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0174887, upper bound: 0.0174887
time: 2.78 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 7.52 seconds
NS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 7.52
Output dim: 1, lower bound: -0.0171544, upper bound: 0.0170843
NS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 7.52
Output dim: 1, lower bound: -0.0171720, upper bound: 0.0170843
NS_A1_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 7.52
Output dim: 1, lower bound: -0.0172541, upper bound: 0.0170924
NS_A1_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 7.52
Output dim: 1, lower bound: -0.0172684, upper bound: 0.0170924
NS_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 7.52
Output dim: 1, lower bound: -0.0172037, upper bound: 0.0174620
NS_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 7.52
Output dim: 1, lower bound: -0.0172037, upper bound: 0.0174786
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 7.52
Output dim: 1, lower bound: -0.0172798, upper bound: 0.0174887
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 7.52
Output dim: 1, lower bound: -0.0172970, upper bound: 0.0174887
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 7.52
Output dim: 1, lower bound: -0.0173699, upper bound: 0.0170843
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 7.52
Output dim: 1, lower bound: -0.0173846, upper bound: 0.0170842
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 7.52
Output dim: 1, lower bound: -0.0174393, upper bound: 0.0170924
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 7.52
Output dim: 1, lower bound: -0.0174540, upper bound: 0.0170924
NS_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 7.52
Output dim: 1, lower bound: -0.0174220, upper bound: 0.0174620
NS_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 7.52
Output dim: 1, lower bound: -0.0174220, upper bound: 0.0174786
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 7.52
Output dim: 1, lower bound: -0.0174730, upper bound: 0.0174887
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 7.52
Output dim: 1, lower bound: -0.0174887, upper bound: 0.0174887

## BFS NS instance: NS_A1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -0.0080320, 0.0067862, -0.0086333, 0.0075732, -0.0156053, 0.0154194
1: 0.9910845, 1.0129638, 0.9906327, 1.0141573, -0.0230728, 0.0223311
2: -0.0102266, 0.0071804, -0.0110125, 0.0078030, -0.0179540, 0.0181022
3: -0.0005342, 0.0050417, -0.0007025, 0.0052291, -0.0057633, 0.0057442
4: -0.0087177, 0.0064996, -0.0095090, 0.0071207, -0.0158384, 0.0160085
5: -0.0028785, 0.0104688, -0.0031523, 0.0114589, -0.0143374, 0.0136211
6: -0.0118317, 0.0021787, -0.0130541, 0.0022754, -0.0141071, 0.0152327
7: -0.0100330, 0.0010728, -0.0103492, 0.0015089, -0.0115419, 0.0114220
8: -0.0148885, 0.0116881, -0.0154980, 0.0127207, -0.0274298, 0.0270437
9: -0.0070745, 0.0083973, -0.0081648, 0.0087549, -0.0158294, 0.0165621

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_A1_A1

### Relational analysis result of NS_A1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0169194, upper bound: 0.0174297
time: 3.12 seconds

## Relational analysis of NS_A1_B2_A1_A1_A2

### Relational analysis result of NS_A1_B2_A1_A1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0169194, upper bound: 0.0171571
time: 2.73 seconds

## BFS NS instance: NS_A1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -0.0081321, 0.0069172, -0.0085247, 0.0074311, -0.0155632, 0.0154419
1: 0.9908620, 1.0131624, 0.9906374, 1.0139419, -0.0230799, 0.0225250
2: -0.0106133, 0.0072840, -0.0110042, 0.0076906, -0.0181577, 0.0181316
3: -0.0005622, 0.0051339, -0.0006721, 0.0052271, -0.0057893, 0.0058060
4: -0.0088494, 0.0068052, -0.0093661, 0.0071141, -0.0159635, 0.0161713
5: -0.0029240, 0.0106336, -0.0031028, 0.0112802, -0.0142042, 0.0137365
6: -0.0120352, 0.0021948, -0.0128334, 0.0022579, -0.0142931, 0.0150281
7: -0.0101886, 0.0011454, -0.0103459, 0.0014301, -0.0116188, 0.0114912
8: -0.0149899, 0.0121962, -0.0153880, 0.0127097, -0.0275235, 0.0274496
9: -0.0072883, 0.0084568, -0.0079679, 0.0086903, -0.0159787, 0.0164248

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_A2_A1

### Relational analysis result of NS_A1_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0169194, upper bound: 0.0174468
time: 3.46 seconds

## Relational analysis of NS_A1_B2_A1_A2_A2

### Relational analysis result of NS_A1_B2_A1_A2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0169194, upper bound: 0.0171845
time: 3.28 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0082482, 0.0070691, -0.0082747, 0.0071039, -0.0153521, 0.0153439
1: 0.9908625, 1.0133927, 0.9905324, 1.0134456, -0.0225831, 0.0228603
2: -0.0106123, 0.0074042, -0.0111870, 0.0074317, -0.0179511, 0.0181898
3: -0.0005947, 0.0051337, -0.0006021, 0.0052707, -0.0058654, 0.0057358
4: -0.0090022, 0.0068044, -0.0090371, 0.0072586, -0.0162608, 0.0158415
5: -0.0029769, 0.0108248, -0.0029890, 0.0108685, -0.0138454, 0.0138138
6: -0.0122712, 0.0022135, -0.0123251, 0.0022177, -0.0144889, 0.0145386
7: -0.0101882, 0.0012296, -0.0104194, 0.0012488, -0.0114370, 0.0116490
8: -0.0151077, 0.0121949, -0.0151345, 0.0129500, -0.0278752, 0.0271824
9: -0.0074665, 0.0085259, -0.0077069, 0.0085417, -0.0160081, 0.0162328

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0169622, upper bound: 0.0174540
time: 3.38 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0169622, upper bound: 0.0172002
time: 3.58 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0081330, 0.0069184, -0.0083676, 0.0072254, -0.0153585, 0.0152860
1: 0.9908673, 1.0131642, 0.9903107, 1.0136300, -0.0227627, 0.0228534
2: -0.0106043, 0.0072850, -0.0115725, 0.0075279, -0.0179763, 0.0184219
3: -0.0005625, 0.0051317, -0.0006281, 0.0053626, -0.0059251, 0.0057599
4: -0.0088506, 0.0067980, -0.0091593, 0.0075633, -0.0164138, 0.0159573
5: -0.0029245, 0.0106352, -0.0030313, 0.0110214, -0.0139458, 0.0136664
6: -0.0120371, 0.0021949, -0.0125139, 0.0022327, -0.0142697, 0.0147088
7: -0.0101850, 0.0011460, -0.0105745, 0.0013162, -0.0115011, 0.0117206
8: -0.0149909, 0.0121843, -0.0152287, 0.0134564, -0.0282702, 0.0272700
9: -0.0072817, 0.0084574, -0.0079881, 0.0085969, -0.0158786, 0.0164455

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0169816, upper bound: 0.0174540
time: 3.12 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0169816, upper bound: 0.0172002
time: 4.06 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0087375, 0.0077097, -0.0088077, 0.0078016, -0.0165391, 0.0165174
1: 0.9911189, 1.0143645, 0.9916216, 1.0145037, -0.0233848, 0.0227429
2: -0.0101666, 0.0079110, -0.0092922, 0.0079837, -0.0181503, 0.0172032
3: -0.0007317, 0.0050274, -0.0007514, 0.0048189, -0.0055506, 0.0057787
4: -0.0096462, 0.0064521, -0.0097386, 0.0057610, -0.0154072, 0.0161907
5: -0.0031998, 0.0116306, -0.0032317, 0.0117462, -0.0149460, 0.0148624
6: -0.0132660, 0.0022922, -0.0134087, 0.0023035, -0.0155695, 0.0157009
7: -0.0100089, 0.0015845, -0.0096571, 0.0016354, -0.0116443, 0.0112416
8: -0.0156037, 0.0116093, -0.0156749, 0.0104604, -0.0259219, 0.0271206
9: -0.0083539, 0.0088169, -0.0084811, 0.0088586, -0.0172125, 0.0172981

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of NS_A2_B1_A1_B1_B1

### Relational analysis result of NS_A2_B1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0171255, upper bound: 0.0168349
time: 3.23 seconds

## Relational analysis of NS_A2_B1_A1_B1_B2

### Relational analysis result of NS_A2_B1_A1_B1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0171953, upper bound: 0.0169152
time: 3.14 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0086415, 0.0075840, -0.0088766, 0.0078918, -0.0165333, 0.0164606
1: 0.9911240, 1.0141739, 0.9913930, 1.0146405, -0.0235165, 0.0227808
2: -0.0101579, 0.0078115, -0.0096898, 0.0080550, -0.0182130, 0.0175013
3: -0.0007048, 0.0050253, -0.0007706, 0.0049137, -0.0056185, 0.0057960
4: -0.0095198, 0.0064453, -0.0098293, 0.0060753, -0.0155951, 0.0162746
5: -0.0031560, 0.0114725, -0.0032631, 0.0118597, -0.0150157, 0.0147356
6: -0.0130708, 0.0022767, -0.0135488, 0.0023146, -0.0153853, 0.0158256
7: -0.0100054, 0.0015148, -0.0098171, 0.0016854, -0.0116908, 0.0113319
8: -0.0155064, 0.0115979, -0.0157448, 0.0109829, -0.0263530, 0.0271816
9: -0.0081797, 0.0087598, -0.0086061, 0.0088996, -0.0170793, 0.0173659

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of NS_A2_B1_A1_B2_B1

### Relational analysis result of NS_A2_B1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0171372, upper bound: 0.0168349
time: 2.84 seconds

## Relational analysis of NS_A2_B1_A1_B2_B2

### Relational analysis result of NS_A2_B1_A1_B2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0172085, upper bound: 0.0169152
time: 2.43 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0085595, 0.0074766, -0.0088505, 0.0078576, -0.0164170, 0.0163271
1: 0.9909206, 1.0140109, 0.9915186, 1.0145888, -0.0236682, 0.0224923
2: -0.0105117, 0.0077266, -0.0094714, 0.0080280, -0.0185397, 0.0171980
3: -0.0006818, 0.0051097, -0.0007633, 0.0048616, -0.0055435, 0.0058730
4: -0.0094118, 0.0067249, -0.0097949, 0.0059027, -0.0153146, 0.0165197
5: -0.0031187, 0.0113374, -0.0032512, 0.0118166, -0.0149353, 0.0145886
6: -0.0129040, 0.0022635, -0.0134956, 0.0023104, -0.0152144, 0.0157592
7: -0.0101477, 0.0014554, -0.0097292, 0.0016664, -0.0118142, 0.0111846
8: -0.0154232, 0.0120627, -0.0157182, 0.0106959, -0.0259779, 0.0276098
9: -0.0080310, 0.0087110, -0.0085587, 0.0088841, -0.0169150, 0.0172697

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of NS_A2_B1_A2_B1_B1

### Relational analysis result of NS_A2_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0171940, upper bound: 0.0168439
time: 2.79 seconds

## Relational analysis of NS_A2_B1_A2_B1_B2

### Relational analysis result of NS_A2_B1_A2_B1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0172602, upper bound: 0.0169227
time: 3.24 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0084508, 0.0073343, -0.0089204, 0.0079491, -0.0163999, 0.0162547
1: 0.9909253, 1.0137953, 0.9912857, 1.0147276, -0.0238023, 0.0225096
2: -0.0105033, 0.0076140, -0.0098767, 0.0081004, -0.0186037, 0.0174907
3: -0.0006514, 0.0051077, -0.0007829, 0.0049583, -0.0056097, 0.0058905
4: -0.0092688, 0.0067182, -0.0098869, 0.0062230, -0.0154918, 0.0166051
5: -0.0030692, 0.0111584, -0.0032831, 0.0119318, -0.0150009, 0.0144414
6: -0.0126830, 0.0022461, -0.0136378, 0.0023216, -0.0150046, 0.0158839
7: -0.0101443, 0.0013765, -0.0098922, 0.0017172, -0.0118615, 0.0112688
8: -0.0153130, 0.0120516, -0.0157891, 0.0112284, -0.0264059, 0.0276713
9: -0.0078338, 0.0086463, -0.0086855, 0.0089257, -0.0167595, 0.0173318

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of NS_A2_B1_A2_B2_B1

### Relational analysis result of NS_A2_B1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0172068, upper bound: 0.0168439
time: 2.73 seconds

## Relational analysis of NS_A2_B1_A2_B2_B2

### Relational analysis result of NS_A2_B1_A2_B2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0172739, upper bound: 0.0169226
time: 2.69 seconds

## BFS NS instance: NS_A2_B2_A1_A1

### Backsubstitution after applying NS history:
0: -0.0084037, 0.0072727, -0.0086503, 0.0075955, -0.0159993, 0.0159230
1: 0.9907166, 1.0137017, 0.9904715, 1.0141913, -0.0234746, 0.0232302
2: -0.0108666, 0.0075653, -0.0112928, 0.0078207, -0.0185070, 0.0186416
3: -0.0006382, 0.0051943, -0.0007073, 0.0052959, -0.0059342, 0.0059016
4: -0.0092068, 0.0070054, -0.0095314, 0.0073422, -0.0165490, 0.0165368
5: -0.0030477, 0.0110809, -0.0031600, 0.0114870, -0.0145347, 0.0142409
6: -0.0125873, 0.0022385, -0.0130887, 0.0022782, -0.0148655, 0.0153272
7: -0.0102905, 0.0013424, -0.0104620, 0.0015212, -0.0118118, 0.0118044
8: -0.0152653, 0.0125290, -0.0155153, 0.0130890, -0.0281716, 0.0278786
9: -0.0077485, 0.0086184, -0.0081957, 0.0087650, -0.0165135, 0.0168141

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A1_A1_A1

### Relational analysis result of NS_A2_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0170604, upper bound: 0.0174297
time: 3.24 seconds

## Relational analysis of NS_A2_B2_A1_A1_A2

### Relational analysis result of NS_A2_B2_A1_A1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0170604, upper bound: 0.0171571
time: 2.59 seconds

## BFS NS instance: NS_A2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -0.0085171, 0.0074212, -0.0085416, 0.0074532, -0.0159703, 0.0159628
1: 0.9904867, 1.0139269, 0.9904763, 1.0139754, -0.0234886, 0.0234506
2: -0.0112662, 0.0076828, -0.0112844, 0.0077081, -0.0187278, 0.0186927
3: -0.0006700, 0.0052896, -0.0006768, 0.0052939, -0.0059639, 0.0059664
4: -0.0093561, 0.0073212, -0.0093883, 0.0073356, -0.0166917, 0.0167095
5: -0.0030994, 0.0112677, -0.0031105, 0.0113079, -0.0144073, 0.0143782
6: -0.0128180, 0.0022567, -0.0128677, 0.0022607, -0.0150786, 0.0151244
7: -0.0104513, 0.0014247, -0.0104586, 0.0014424, -0.0118937, 0.0118833
8: -0.0153803, 0.0130541, -0.0154051, 0.0130780, -0.0282813, 0.0282967
9: -0.0079542, 0.0086858, -0.0079985, 0.0087004, -0.0166546, 0.0166844

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A1_A2_A1

### Relational analysis result of NS_A2_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0170604, upper bound: 0.0174468
time: 3.03 seconds

## Relational analysis of NS_A2_B2_A1_A2_A2

### Relational analysis result of NS_A2_B2_A1_A2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0170604, upper bound: 0.0171845
time: 3.05 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0086376, 0.0075789, -0.0082920, 0.0071265, -0.0157640, 0.0158709
1: 0.9904920, 1.0141659, 0.9903715, 1.0134797, -0.0229877, 0.0237944
2: -0.0112570, 0.0078075, -0.0114670, 0.0074496, -0.0185167, 0.0188029
3: -0.0007037, 0.0052874, -0.0006070, 0.0053375, -0.0060412, 0.0058944
4: -0.0095146, 0.0073140, -0.0090598, 0.0074799, -0.0169945, 0.0163738
5: -0.0031543, 0.0114660, -0.0029969, 0.0108969, -0.0140512, 0.0144629
6: -0.0130628, 0.0022761, -0.0123602, 0.0022205, -0.0152833, 0.0146363
7: -0.0104476, 0.0015120, -0.0105321, 0.0012613, -0.0117089, 0.0120441
8: -0.0155024, 0.0130420, -0.0151521, 0.0133179, -0.0286369, 0.0280188
9: -0.0081726, 0.0087575, -0.0079112, 0.0085519, -0.0167245, 0.0166686

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0170760, upper bound: 0.0174540
time: 3.47 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0170760, upper bound: 0.0172002
time: 2.90 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0085282, 0.0074357, -0.0083848, 0.0072479, -0.0157761, 0.0158205
1: 0.9904969, 1.0139487, 0.9901464, 1.0136641, -0.0231672, 0.0238023
2: -0.0112487, 0.0076942, -0.0118583, 0.0075456, -0.0185465, 0.0190836
3: -0.0006731, 0.0052854, -0.0006329, 0.0054308, -0.0061038, 0.0059183
4: -0.0093707, 0.0073073, -0.0091819, 0.0077891, -0.0171598, 0.0164892
5: -0.0031045, 0.0112859, -0.0030391, 0.0110497, -0.0141542, 0.0143250
6: -0.0128405, 0.0022585, -0.0125488, 0.0022354, -0.0150759, 0.0148073
7: -0.0104442, 0.0014327, -0.0106895, 0.0013286, -0.0117728, 0.0121222
8: -0.0153915, 0.0130310, -0.0152461, 0.0138320, -0.0290480, 0.0281060
9: -0.0079743, 0.0086924, -0.0081967, 0.0086071, -0.0165814, 0.0168891

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0170924, upper bound: 0.0174540
time: 2.75 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0170924, upper bound: 0.0172002
time: 2.97 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 7.31 seconds
NS_A1_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 7.31
Output dim: 1, lower bound: -0.0169194, upper bound: 0.0174297
NS_A1_B2_A1_A1_A2, status: Status.VERIFIED, split count: 5, time: 7.31
Output dim: 1, lower bound: -0.0169194, upper bound: 0.0171571
NS_A1_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 7.31
Output dim: 1, lower bound: -0.0169194, upper bound: 0.0174468
NS_A1_B2_A1_A2_A2, status: Status.VERIFIED, split count: 5, time: 7.31
Output dim: 1, lower bound: -0.0169194, upper bound: 0.0171845
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.31
Output dim: 1, lower bound: -0.0169622, upper bound: 0.0174540
NS_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 7.31
Output dim: 1, lower bound: -0.0169622, upper bound: 0.0172002
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.31
Output dim: 1, lower bound: -0.0169816, upper bound: 0.0174540
NS_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 7.31
Output dim: 1, lower bound: -0.0169816, upper bound: 0.0172002
NS_A2_B1_A1_B1_B1, status: Status.VERIFIED, split count: 5, time: 7.31
Output dim: 1, lower bound: -0.0171255, upper bound: 0.0168349
NS_A2_B1_A1_B1_B2, status: Status.VERIFIED, split count: 5, time: 7.31
Output dim: 1, lower bound: -0.0171953, upper bound: 0.0169152
NS_A2_B1_A1_B2_B1, status: Status.VERIFIED, split count: 5, time: 7.31
Output dim: 1, lower bound: -0.0171372, upper bound: 0.0168349
NS_A2_B1_A1_B2_B2, status: Status.VERIFIED, split count: 5, time: 7.31
Output dim: 1, lower bound: -0.0172085, upper bound: 0.0169152
NS_A2_B1_A2_B1_B1, status: Status.VERIFIED, split count: 5, time: 7.31
Output dim: 1, lower bound: -0.0171940, upper bound: 0.0168439
NS_A2_B1_A2_B1_B2, status: Status.VERIFIED, split count: 5, time: 7.31
Output dim: 1, lower bound: -0.0172602, upper bound: 0.0169227
NS_A2_B1_A2_B2_B1, status: Status.VERIFIED, split count: 5, time: 7.31
Output dim: 1, lower bound: -0.0172068, upper bound: 0.0168439
NS_A2_B1_A2_B2_B2, status: Status.VERIFIED, split count: 5, time: 7.31
Output dim: 1, lower bound: -0.0172739, upper bound: 0.0169226
NS_A2_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 7.31
Output dim: 1, lower bound: -0.0170604, upper bound: 0.0174297
NS_A2_B2_A1_A1_A2, status: Status.VERIFIED, split count: 5, time: 7.31
Output dim: 1, lower bound: -0.0170604, upper bound: 0.0171571
NS_A2_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 7.31
Output dim: 1, lower bound: -0.0170604, upper bound: 0.0174468
NS_A2_B2_A1_A2_A2, status: Status.VERIFIED, split count: 5, time: 7.31
Output dim: 1, lower bound: -0.0170604, upper bound: 0.0171845
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.31
Output dim: 1, lower bound: -0.0170760, upper bound: 0.0174540
NS_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 7.31
Output dim: 1, lower bound: -0.0170760, upper bound: 0.0172002
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.31
Output dim: 1, lower bound: -0.0170924, upper bound: 0.0174540
NS_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 7.31
Output dim: 1, lower bound: -0.0170924, upper bound: 0.0172002

## BFS NS instance: NS_A1_B2_A1_A1_A1

### Backsubstitution after applying NS history:
0: -0.0085937, 0.0075214, -0.0086333, 0.0075732, -0.0161669, 0.0161547
1: 0.9923046, 1.0140789, 0.9906327, 1.0141573, -0.0218527, 0.0234462
2: -0.0084852, 0.0077620, -0.0110125, 0.0078030, -0.0162882, 0.0187651
3: -0.0006914, 0.0045356, -0.0007025, 0.0052291, -0.0059205, 0.0052381
4: -0.0094569, 0.0048220, -0.0095090, 0.0071207, -0.0165775, 0.0143309
5: -0.0031343, 0.0113937, -0.0031523, 0.0114589, -0.0145932, 0.0145460
6: -0.0129735, 0.0022691, -0.0130541, 0.0022754, -0.0152489, 0.0153231
7: -0.0091791, 0.0014802, -0.0103492, 0.0015089, -0.0106880, 0.0118294
8: -0.0154579, 0.0088992, -0.0154980, 0.0127207, -0.0280401, 0.0242566
9: -0.0080930, 0.0087313, -0.0081648, 0.0087549, -0.0168479, 0.0168961

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 64

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 136

## Relational analysis of NS_A1_B2_A1_A1_A1_A1

### Relational analysis result of NS_A1_B2_A1_A1_A1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0166838, upper bound: 0.0171849
time: 2.92 seconds

## Relational analysis of NS_A1_B2_A1_A1_A1_A2

### Relational analysis result of NS_A1_B2_A1_A1_A1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0167541, upper bound: 0.0172515
time: 3.19 seconds

## BFS NS instance: NS_A1_B2_A1_A2_A1

### Backsubstitution after applying NS history:
0: -0.0086739, 0.0076265, -0.0085247, 0.0074311, -0.0161050, 0.0161512
1: 0.9920655, 1.0142381, 0.9906374, 1.0139419, -0.0218763, 0.0236007
2: -0.0085895, 0.0078451, -0.0110042, 0.0076906, -0.0162801, 0.0187838
3: -0.0007139, 0.0046348, -0.0006721, 0.0052271, -0.0059410, 0.0053069
4: -0.0095625, 0.0051507, -0.0093661, 0.0071141, -0.0166766, 0.0145168
5: -0.0031708, 0.0115259, -0.0031028, 0.0112802, -0.0144510, 0.0146287
6: -0.0131367, 0.0022820, -0.0128334, 0.0022579, -0.0153946, 0.0151153
7: -0.0093464, 0.0015384, -0.0103459, 0.0014301, -0.0107766, 0.0118843
8: -0.0155393, 0.0094458, -0.0153880, 0.0127097, -0.0281126, 0.0246984
9: -0.0082385, 0.0087791, -0.0079679, 0.0086903, -0.0169288, 0.0167470

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 137

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 136

## Relational analysis of NS_A1_B2_A1_A2_A1_A1

### Relational analysis result of NS_A1_B2_A1_A2_A1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0166838, upper bound: 0.0171989
time: 2.87 seconds

## Relational analysis of NS_A1_B2_A1_A2_A1_A2

### Relational analysis result of NS_A1_B2_A1_A2_A1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0167541, upper bound: 0.0172658
time: 3.37 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0087824, 0.0077685, -0.0082747, 0.0071039, -0.0158863, 0.0160432
1: 0.9921122, 1.0144535, 0.9905324, 1.0134456, -0.0213334, 0.0239211
2: -0.0087306, 0.0079574, -0.0111870, 0.0074317, -0.0161623, 0.0188560
3: -0.0007443, 0.0046154, -0.0006021, 0.0052707, -0.0060149, 0.0052175
4: -0.0097053, 0.0050866, -0.0090371, 0.0072586, -0.0169639, 0.0141237
5: -0.0032202, 0.0117045, -0.0029890, 0.0108685, -0.0140887, 0.0146935
6: -0.0133572, 0.0022994, -0.0123251, 0.0022177, -0.0155750, 0.0146246
7: -0.0093138, 0.0016171, -0.0104194, 0.0012488, -0.0105626, 0.0120365
8: -0.0156492, 0.0093393, -0.0151345, 0.0129500, -0.0284598, 0.0243271
9: -0.0084352, 0.0088436, -0.0077069, 0.0085417, -0.0169769, 0.0165505

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 64

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 136

## Relational analysis of NS_A1_B2_A2_B1_A1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0167177, upper bound: 0.0172068
time: 3.12 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0167976, upper bound: 0.0172739
time: 3.30 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0086746, 0.0076274, -0.0083676, 0.0072254, -0.0159001, 0.0159950
1: 0.9921168, 1.0142397, 0.9903107, 1.0136300, -0.0215132, 0.0239289
2: -0.0085905, 0.0078459, -0.0115725, 0.0075279, -0.0161183, 0.0190958
3: -0.0007141, 0.0046135, -0.0006281, 0.0053626, -0.0060767, 0.0052416
4: -0.0095634, 0.0050802, -0.0091593, 0.0075633, -0.0171267, 0.0142394
5: -0.0031711, 0.0115271, -0.0030313, 0.0110214, -0.0141925, 0.0145584
6: -0.0131382, 0.0022821, -0.0125139, 0.0022327, -0.0153708, 0.0147960
7: -0.0093105, 0.0015389, -0.0105745, 0.0013162, -0.0106267, 0.0121134
8: -0.0155400, 0.0093285, -0.0152287, 0.0134564, -0.0288609, 0.0244145
9: -0.0082398, 0.0087795, -0.0079881, 0.0085969, -0.0168367, 0.0167676

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 64

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 136

## Relational analysis of NS_A1_B2_A2_B2_A1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0167390, upper bound: 0.0172068
time: 2.78 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0168168, upper bound: 0.0172739
time: 3.37 seconds

## BFS NS instance: NS_A2_B2_A1_A1_A1

### Backsubstitution after applying NS history:
0: -0.0089618, 0.0080033, -0.0086503, 0.0075955, -0.0165573, 0.0166536
1: 0.9919350, 1.0148098, 0.9904715, 1.0141913, -0.0222563, 0.0243384
2: -0.0089638, 0.0081432, -0.0112928, 0.0078207, -0.0167845, 0.0193218
3: -0.0007945, 0.0046889, -0.0007073, 0.0052959, -0.0060904, 0.0053961
4: -0.0099414, 0.0053301, -0.0095314, 0.0073422, -0.0172836, 0.0148615
5: -0.0033019, 0.0119999, -0.0031600, 0.0114870, -0.0147889, 0.0151600
6: -0.0137219, 0.0023283, -0.0130887, 0.0022782, -0.0160001, 0.0154170
7: -0.0094377, 0.0017472, -0.0104620, 0.0015212, -0.0109590, 0.0122092
8: -0.0158311, 0.0097440, -0.0155153, 0.0130890, -0.0287793, 0.0250916
9: -0.0087605, 0.0089503, -0.0081957, 0.0087650, -0.0175256, 0.0171460

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 64

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 136

## Relational analysis of NS_A2_B2_A1_A1_A1_A1

### Relational analysis result of NS_A2_B2_A1_A1_A1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0168069, upper bound: 0.0171849
time: 3.21 seconds

## Relational analysis of NS_A2_B2_A1_A1_A1_A2

### Relational analysis result of NS_A2_B2_A1_A1_A1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0168901, upper bound: 0.0172515
time: 3.22 seconds

## BFS NS instance: NS_A2_B2_A1_A2_A1

### Backsubstitution after applying NS history:
0: -0.0090411, 0.0081071, -0.0085416, 0.0074532, -0.0164943, 0.0166487
1: 0.9917047, 1.0149672, 0.9904763, 1.0139754, -0.0222707, 0.0244910
2: -0.0091478, 0.0082254, -0.0112844, 0.0077081, -0.0168558, 0.0193383
3: -0.0008167, 0.0047844, -0.0006768, 0.0052939, -0.0061106, 0.0054613
4: -0.0100457, 0.0056469, -0.0093883, 0.0073356, -0.0173813, 0.0150352
5: -0.0033380, 0.0121305, -0.0031105, 0.0113079, -0.0146460, 0.0152411
6: -0.0138832, 0.0023410, -0.0128677, 0.0022607, -0.0161438, 0.0152087
7: -0.0095990, 0.0018047, -0.0104586, 0.0014424, -0.0110414, 0.0122633
8: -0.0159115, 0.0102707, -0.0154051, 0.0130780, -0.0288510, 0.0255160
9: -0.0089044, 0.0089974, -0.0079985, 0.0087004, -0.0176047, 0.0169960

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 64

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 136

## Relational analysis of NS_A2_B2_A1_A2_A1_A1

### Relational analysis result of NS_A2_B2_A1_A2_A1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0168069, upper bound: 0.0171989
time: 3.35 seconds

## Relational analysis of NS_A2_B2_A1_A2_A1_A2

### Relational analysis result of NS_A2_B2_A1_A2_A1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0168899, upper bound: 0.0172658
time: 3.29 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0091577, 0.0082597, -0.0082920, 0.0071265, -0.0162841, 0.0165517
1: 0.9917466, 1.0151988, 0.9903715, 1.0134797, -0.0217331, 0.0248274
2: -0.0092185, 0.0083461, -0.0114670, 0.0074496, -0.0166681, 0.0194561
3: -0.0008493, 0.0047670, -0.0006070, 0.0053375, -0.0061868, 0.0053740
4: -0.0101992, 0.0055891, -0.0090598, 0.0074799, -0.0176791, 0.0146489
5: -0.0033911, 0.0123225, -0.0029969, 0.0108969, -0.0142881, 0.0153194
6: -0.0141202, 0.0023598, -0.0123602, 0.0022205, -0.0163407, 0.0147200
7: -0.0095696, 0.0018893, -0.0105321, 0.0012613, -0.0108309, 0.0124214
8: -0.0160297, 0.0101746, -0.0151521, 0.0133179, -0.0292064, 0.0251540
9: -0.0091158, 0.0090668, -0.0079112, 0.0085519, -0.0176677, 0.0169780

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 64

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 136

## Relational analysis of NS_A2_B2_A2_B1_A1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0168277, upper bound: 0.0172068
time: 3.29 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0169071, upper bound: 0.0172739
time: 3.16 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0090484, 0.0081168, -0.0083848, 0.0072479, -0.0162964, 0.0165016
1: 0.9917513, 1.0149819, 0.9901464, 1.0136641, -0.0219128, 0.0248355
2: -0.0090765, 0.0082330, -0.0118583, 0.0075456, -0.0166222, 0.0197374
3: -0.0008187, 0.0047650, -0.0006329, 0.0054308, -0.0062495, 0.0053980
4: -0.0100555, 0.0055826, -0.0091819, 0.0077891, -0.0178446, 0.0147645
5: -0.0033414, 0.0121427, -0.0030391, 0.0110497, -0.0143911, 0.0151818
6: -0.0138982, 0.0023422, -0.0125488, 0.0022354, -0.0161336, 0.0148910
7: -0.0095663, 0.0018101, -0.0106895, 0.0013286, -0.0108949, 0.0124995
8: -0.0159190, 0.0101637, -0.0152461, 0.0138320, -0.0296134, 0.0252414
9: -0.0089177, 0.0090018, -0.0081967, 0.0086071, -0.0175248, 0.0171985

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 64

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 136

## Relational analysis of NS_A2_B2_A2_B2_A1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0168439, upper bound: 0.0172068
time: 3.10 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0169226, upper bound: 0.0172739
time: 2.87 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 8.02 seconds
NS_A1_B2_A1_A1_A1_A1, status: Status.VERIFIED, split count: 6, time: 8.02
Output dim: 1, lower bound: -0.0166838, upper bound: 0.0171849
NS_A1_B2_A1_A1_A1_A2, status: Status.VERIFIED, split count: 6, time: 8.02
Output dim: 1, lower bound: -0.0167541, upper bound: 0.0172515
NS_A1_B2_A1_A2_A1_A1, status: Status.VERIFIED, split count: 6, time: 8.02
Output dim: 1, lower bound: -0.0166838, upper bound: 0.0171989
NS_A1_B2_A1_A2_A1_A2, status: Status.VERIFIED, split count: 6, time: 8.02
Output dim: 1, lower bound: -0.0167541, upper bound: 0.0172658
NS_A1_B2_A2_B1_A1_A1, status: Status.VERIFIED, split count: 6, time: 8.02
Output dim: 1, lower bound: -0.0167177, upper bound: 0.0172068
NS_A1_B2_A2_B1_A1_A2, status: Status.VERIFIED, split count: 6, time: 8.02
Output dim: 1, lower bound: -0.0167976, upper bound: 0.0172739
NS_A1_B2_A2_B2_A1_A1, status: Status.VERIFIED, split count: 6, time: 8.02
Output dim: 1, lower bound: -0.0167390, upper bound: 0.0172068
NS_A1_B2_A2_B2_A1_A2, status: Status.VERIFIED, split count: 6, time: 8.02
Output dim: 1, lower bound: -0.0168168, upper bound: 0.0172739
NS_A2_B2_A1_A1_A1_A1, status: Status.VERIFIED, split count: 6, time: 8.02
Output dim: 1, lower bound: -0.0168069, upper bound: 0.0171849
NS_A2_B2_A1_A1_A1_A2, status: Status.VERIFIED, split count: 6, time: 8.02
Output dim: 1, lower bound: -0.0168901, upper bound: 0.0172515
NS_A2_B2_A1_A2_A1_A1, status: Status.VERIFIED, split count: 6, time: 8.02
Output dim: 1, lower bound: -0.0168069, upper bound: 0.0171989
NS_A2_B2_A1_A2_A1_A2, status: Status.VERIFIED, split count: 6, time: 8.02
Output dim: 1, lower bound: -0.0168899, upper bound: 0.0172658
NS_A2_B2_A2_B1_A1_A1, status: Status.VERIFIED, split count: 6, time: 8.02
Output dim: 1, lower bound: -0.0168277, upper bound: 0.0172068
NS_A2_B2_A2_B1_A1_A2, status: Status.VERIFIED, split count: 6, time: 8.02
Output dim: 1, lower bound: -0.0169071, upper bound: 0.0172739
NS_A2_B2_A2_B2_A1_A1, status: Status.VERIFIED, split count: 6, time: 8.02
Output dim: 1, lower bound: -0.0168439, upper bound: 0.0172068
NS_A2_B2_A2_B2_A1_A2, status: Status.VERIFIED, split count: 6, time: 8.02
Output dim: 1, lower bound: -0.0169226, upper bound: 0.0172739

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 5.14 + 273.80 = 278.94 seconds
