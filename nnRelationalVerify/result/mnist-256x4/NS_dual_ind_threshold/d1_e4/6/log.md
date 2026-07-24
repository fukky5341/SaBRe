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
execution time: IAR + RelationalAnalysis = 1.60 + 3.46 = 5.06 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.0180128, upper bound: 0.0180128

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 137

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 59

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0177198, upper bound: 0.0179069
time: 2.56 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0179069, upper bound: 0.0179069
time: 3.46 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 6.15 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 6.15
Output dim: 1, lower bound: -0.0177198, upper bound: 0.0179069
NS_A2, status: Status.UNKNOWN, split count: 1, time: 6.15
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

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 137

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0175767, upper bound: 0.0174265
time: 2.97 seconds

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

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 137

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0177684, upper bound: 0.0174265
time: 2.61 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0177991, upper bound: 0.0177991
time: 3.32 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 7.46 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 7.46
Output dim: 1, lower bound: -0.0175767, upper bound: 0.0174265
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 7.46
Output dim: 1, lower bound: -0.0176041, upper bound: 0.0177991
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 7.46
Output dim: 1, lower bound: -0.0177684, upper bound: 0.0174265
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 7.46
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

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 137

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0173860, upper bound: 0.0173031
time: 3.22 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0174820, upper bound: 0.0173130
time: 2.77 seconds

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

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 137

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0172969, upper bound: 0.0177684
time: 3.36 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0172969, upper bound: 0.0177991
time: 3.36 seconds

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

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 137

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0176008, upper bound: 0.0173031
time: 3.05 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0176671, upper bound: 0.0173130
time: 3.26 seconds

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

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 137

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0174265, upper bound: 0.0177684
time: 2.81 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0174265, upper bound: 0.0177992
time: 2.69 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 7.45 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 7.45
Output dim: 1, lower bound: -0.0173860, upper bound: 0.0173031
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 7.45
Output dim: 1, lower bound: -0.0174820, upper bound: 0.0173130
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 7.45
Output dim: 1, lower bound: -0.0172969, upper bound: 0.0177684
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 7.45
Output dim: 1, lower bound: -0.0172969, upper bound: 0.0177991
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 7.45
Output dim: 1, lower bound: -0.0176008, upper bound: 0.0173031
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 7.45
Output dim: 1, lower bound: -0.0176671, upper bound: 0.0173130
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 7.45
Output dim: 1, lower bound: -0.0174265, upper bound: 0.0177684
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 7.45
Output dim: 1, lower bound: -0.0174265, upper bound: 0.0177992

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

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 137

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0171544, upper bound: 0.0170843
time: 3.34 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0171720, upper bound: 0.0170843
time: 3.22 seconds

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

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 137

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0172541, upper bound: 0.0170924
time: 3.05 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0172684, upper bound: 0.0170924
time: 3.45 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0090468, 0.0081146, -0.0088899, 0.0079092, -0.0169560, 0.0170045
1: 0.9919129, 1.0149785, 0.9904547, 1.0146670, -0.0227541, 0.0245238
2: -0.0090743, 0.0082313, -0.0113222, 0.0080688, -0.0171431, 0.0195534
3: -0.0008183, 0.0046981, -0.0007744, 0.0053029, -0.0061212, 0.0054724
4: -0.0100532, 0.0053607, -0.0098468, 0.0073654, -0.0174187, 0.0152074
5: -0.0033406, 0.0121399, -0.0032692, 0.0118816, -0.0152222, 0.0154091
6: -0.0138948, 0.0023419, -0.0135758, 0.0023167, -0.0162115, 0.0159178
7: -0.0094533, 0.0018088, -0.0104738, 0.0016951, -0.0111483, 0.0122826
8: -0.0159173, 0.0097948, -0.0157582, 0.0131276, -0.0289039, 0.0254121
9: -0.0089147, 0.0090008, -0.0086302, 0.0089075, -0.0178222, 0.0176310

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 137

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0171863, upper bound: 0.0176008
time: 2.97 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0171962, upper bound: 0.0176671
time: 3.31 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0084797, 0.0073722, -0.0088899, 0.0079092, -0.0163889, 0.0162621
1: 0.9907717, 1.0138526, 0.9904547, 1.0146670, -0.0238954, 0.0233979
2: -0.0107706, 0.0076440, -0.0113222, 0.0080688, -0.0187737, 0.0189397
3: -0.0006595, 0.0051714, -0.0007744, 0.0053029, -0.0059624, 0.0059458
4: -0.0093069, 0.0069295, -0.0098468, 0.0073654, -0.0166723, 0.0167763
5: -0.0030824, 0.0112060, -0.0032692, 0.0118816, -0.0149639, 0.0144752
6: -0.0127418, 0.0022507, -0.0135758, 0.0023167, -0.0150585, 0.0158265
7: -0.0102519, 0.0013975, -0.0104738, 0.0016951, -0.0119470, 0.0118713
8: -0.0153424, 0.0124029, -0.0157582, 0.0131276, -0.0282880, 0.0279820
9: -0.0078863, 0.0086636, -0.0086302, 0.0089075, -0.0167938, 0.0172938

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 137

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0171863, upper bound: 0.0173724
time: 3.04 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0171962, upper bound: 0.0174075
time: 3.11 seconds

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

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 137

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0173699, upper bound: 0.0170843
time: 2.72 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0173846, upper bound: 0.0170842
time: 3.25 seconds

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

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 137

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0174393, upper bound: 0.0170924
time: 2.83 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0174540, upper bound: 0.0170924
time: 3.04 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0094218, 0.0086055, -0.0089071, 0.0079317, -0.0173535, 0.0175126
1: 0.9915416, 1.0157232, 0.9902934, 1.0147012, -0.0231596, 0.0254298
2: -0.0095620, 0.0086196, -0.0116026, 0.0080866, -0.0176486, 0.0201528
3: -0.0009233, 0.0048521, -0.0007792, 0.0053698, -0.0062930, 0.0056312
4: -0.0105468, 0.0058710, -0.0098694, 0.0075870, -0.0181339, 0.0157404
5: -0.0035114, 0.0127575, -0.0032770, 0.0119099, -0.0154213, 0.0160345
6: -0.0146572, 0.0024023, -0.0136108, 0.0023195, -0.0169767, 0.0160131
7: -0.0097131, 0.0020809, -0.0105866, 0.0017075, -0.0114206, 0.0126675
8: -0.0162975, 0.0106432, -0.0157757, 0.0134960, -0.0296510, 0.0262515
9: -0.0095948, 0.0092239, -0.0086614, 0.0089178, -0.0185126, 0.0178853

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 137

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0173031, upper bound: 0.0176008
time: 3.60 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0173130, upper bound: 0.0176671
time: 2.98 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0088695, 0.0078825, -0.0089071, 0.0079317, -0.0168012, 0.0167896
1: 0.9903982, 1.0146267, 0.9902934, 1.0147012, -0.0243030, 0.0243334
2: -0.0114204, 0.0080476, -0.0116026, 0.0080866, -0.0193423, 0.0194905
3: -0.0007686, 0.0053263, -0.0007792, 0.0053698, -0.0061384, 0.0061055
4: -0.0098199, 0.0074430, -0.0098694, 0.0075870, -0.0174069, 0.0173125
5: -0.0032599, 0.0118479, -0.0032770, 0.0119099, -0.0151698, 0.0151249
6: -0.0135343, 0.0023134, -0.0136108, 0.0023195, -0.0158538, 0.0159242
7: -0.0105133, 0.0016802, -0.0105866, 0.0017075, -0.0122208, 0.0122668
8: -0.0157375, 0.0132566, -0.0157757, 0.0134960, -0.0290508, 0.0288252
9: -0.0085932, 0.0088954, -0.0086614, 0.0089178, -0.0175109, 0.0175568

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 137

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0173031, upper bound: 0.0173724
time: 2.92 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0173130, upper bound: 0.0174075
time: 3.49 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 7.94 seconds
NS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 7.94
Output dim: 1, lower bound: -0.0171544, upper bound: 0.0170843
NS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 7.94
Output dim: 1, lower bound: -0.0171720, upper bound: 0.0170843
NS_A1_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 7.94
Output dim: 1, lower bound: -0.0172541, upper bound: 0.0170924
NS_A1_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 7.94
Output dim: 1, lower bound: -0.0172684, upper bound: 0.0170924
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 7.94
Output dim: 1, lower bound: -0.0171863, upper bound: 0.0176008
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 7.94
Output dim: 1, lower bound: -0.0171962, upper bound: 0.0176671
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 7.94
Output dim: 1, lower bound: -0.0171863, upper bound: 0.0173724
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 7.94
Output dim: 1, lower bound: -0.0171962, upper bound: 0.0174075
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 7.94
Output dim: 1, lower bound: -0.0173699, upper bound: 0.0170843
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 7.94
Output dim: 1, lower bound: -0.0173846, upper bound: 0.0170842
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 7.94
Output dim: 1, lower bound: -0.0174393, upper bound: 0.0170924
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 7.94
Output dim: 1, lower bound: -0.0174540, upper bound: 0.0170924
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 7.94
Output dim: 1, lower bound: -0.0173031, upper bound: 0.0176008
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 7.94
Output dim: 1, lower bound: -0.0173130, upper bound: 0.0176671
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 7.94
Output dim: 1, lower bound: -0.0173031, upper bound: 0.0173724
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 7.94
Output dim: 1, lower bound: -0.0173130, upper bound: 0.0174075

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0089822, 0.0080301, -0.0089870, 0.0080363, -0.0170185, 0.0170171
1: 0.9920739, 1.0148503, 0.9908516, 1.0148600, -0.0227861, 0.0239986
2: -0.0089904, 0.0081644, -0.0106315, 0.0081694, -0.0171597, 0.0187959
3: -0.0008002, 0.0046313, -0.0008015, 0.0051382, -0.0059384, 0.0054328
4: -0.0099682, 0.0051393, -0.0099746, 0.0068196, -0.0167878, 0.0151139
5: -0.0033112, 0.0120336, -0.0033134, 0.0120415, -0.0153527, 0.0153470
6: -0.0137635, 0.0023316, -0.0137732, 0.0023323, -0.0160958, 0.0161048
7: -0.0093406, 0.0017620, -0.0101959, 0.0017655, -0.0111061, 0.0119579
8: -0.0158518, 0.0094268, -0.0158567, 0.0122201, -0.0279329, 0.0251425
9: -0.0087976, 0.0089624, -0.0088063, 0.0089653, -0.0177629, 0.0177687

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 137

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0169733, upper bound: 0.0173699
time: 3.26 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0169733, upper bound: 0.0173846
time: 2.43 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0090250, 0.0080861, -0.0088046, 0.0077975, -0.0168225, 0.0168906
1: 0.9919668, 1.0149353, 0.9906455, 1.0144975, -0.0225307, 0.0242897
2: -0.0090460, 0.0082087, -0.0109900, 0.0079804, -0.0170264, 0.0191987
3: -0.0008122, 0.0046757, -0.0007505, 0.0052237, -0.0060359, 0.0054262
4: -0.0100246, 0.0052866, -0.0097345, 0.0071029, -0.0171275, 0.0150211
5: -0.0033307, 0.0121040, -0.0032303, 0.0117410, -0.0150717, 0.0153343
6: -0.0138505, 0.0023384, -0.0134023, 0.0023030, -0.0161535, 0.0157408
7: -0.0094156, 0.0017930, -0.0103402, 0.0016332, -0.0110488, 0.0121332
8: -0.0158952, 0.0096717, -0.0156717, 0.0126912, -0.0284427, 0.0252029
9: -0.0088752, 0.0089879, -0.0084755, 0.0088568, -0.0177320, 0.0174633

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 137

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0169816, upper bound: 0.0174393
time: 3.35 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0169816, upper bound: 0.0174540
time: 2.91 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0084090, 0.0072796, -0.0089870, 0.0080363, -0.0164453, 0.0162666
1: 0.9909413, 1.0137120, 0.9908516, 1.0148600, -0.0239187, 0.0228604
2: -0.0104755, 0.0075707, -0.0106315, 0.0081694, -0.0186448, 0.0182023
3: -0.0006397, 0.0051010, -0.0008015, 0.0051382, -0.0057780, 0.0059026
4: -0.0092138, 0.0066963, -0.0099746, 0.0068196, -0.0160334, 0.0166708
5: -0.0030501, 0.0110896, -0.0033134, 0.0120415, -0.0150916, 0.0144030
6: -0.0125981, 0.0022393, -0.0137732, 0.0023323, -0.0149304, 0.0160126
7: -0.0101332, 0.0013462, -0.0101959, 0.0017655, -0.0118987, 0.0115421
8: -0.0152707, 0.0120151, -0.0158567, 0.0122201, -0.0273109, 0.0276945
9: -0.0077581, 0.0086215, -0.0088063, 0.0089653, -0.0167234, 0.0174278

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 137

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0172862, upper bound: 0.0171312
time: 3.14 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0172862, upper bound: 0.0171597
time: 2.80 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0084554, 0.0073404, -0.0088046, 0.0077975, -0.0162529, 0.0161450
1: 0.9908249, 1.0138043, 0.9906455, 1.0144975, -0.0236726, 0.0231588
2: -0.0106780, 0.0076188, -0.0109900, 0.0079804, -0.0186123, 0.0186089
3: -0.0006527, 0.0051493, -0.0007505, 0.0052237, -0.0058765, 0.0058998
4: -0.0092749, 0.0068563, -0.0097345, 0.0071029, -0.0163779, 0.0165908
5: -0.0030713, 0.0111661, -0.0032303, 0.0117410, -0.0148123, 0.0143964
6: -0.0126925, 0.0022468, -0.0134023, 0.0023030, -0.0149955, 0.0156491
7: -0.0102146, 0.0013799, -0.0103402, 0.0016332, -0.0118478, 0.0117201
8: -0.0153177, 0.0122812, -0.0156717, 0.0126912, -0.0278213, 0.0277739
9: -0.0078423, 0.0086491, -0.0084755, 0.0088568, -0.0166991, 0.0171246

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 137

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0172970, upper bound: 0.0171737
time: 3.30 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0172970, upper bound: 0.0172002
time: 3.37 seconds

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

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 137

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0170465, upper bound: 0.0170843
time: 2.54 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0170465, upper bound: 0.0170843
time: 3.24 seconds

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

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 137

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0170654, upper bound: 0.0170843
time: 3.32 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0170654, upper bound: 0.0170843
time: 3.24 seconds

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

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 137

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0170848, upper bound: 0.0170924
time: 2.58 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0170848, upper bound: 0.0170924
time: 2.45 seconds

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

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 137

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0171000, upper bound: 0.0170924
time: 3.13 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0171000, upper bound: 0.0170924
time: 2.78 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0093573, 0.0085212, -0.0090036, 0.0080580, -0.0174154, 0.0175247
1: 0.9917019, 1.0155952, 0.9906914, 1.0148927, -0.0231908, 0.0249038
2: -0.0094781, 0.0085529, -0.0109103, 0.0081865, -0.0176647, 0.0194632
3: -0.0009052, 0.0047856, -0.0008062, 0.0052047, -0.0061099, 0.0055917
4: -0.0104620, 0.0056506, -0.0099964, 0.0070399, -0.0175019, 0.0156470
5: -0.0034821, 0.0126514, -0.0033210, 0.0120688, -0.0155509, 0.0159723
6: -0.0145262, 0.0023919, -0.0138069, 0.0023350, -0.0168612, 0.0161989
7: -0.0096009, 0.0020341, -0.0103081, 0.0017775, -0.0113784, 0.0123422
8: -0.0162321, 0.0102768, -0.0158735, 0.0125864, -0.0286780, 0.0259840
9: -0.0094779, 0.0091855, -0.0088364, 0.0089751, -0.0184530, 0.0180219

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 137

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0170843, upper bound: 0.0173699
time: 3.21 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0170843, upper bound: 0.0173846
time: 3.30 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0094004, 0.0085776, -0.0088219, 0.0078202, -0.0172206, 0.0173994
1: 0.9915964, 1.0156808, 0.9904856, 1.0145320, -0.0229356, 0.0251952
2: -0.0095341, 0.0085975, -0.0112683, 0.0079984, -0.0175325, 0.0198658
3: -0.0009173, 0.0048293, -0.0007553, 0.0052901, -0.0062074, 0.0055846
4: -0.0105187, 0.0057956, -0.0097572, 0.0073229, -0.0178416, 0.0155529
5: -0.0035017, 0.0127223, -0.0032382, 0.0117696, -0.0152713, 0.0159605
6: -0.0146138, 0.0023989, -0.0134376, 0.0023058, -0.0169195, 0.0158364
7: -0.0096747, 0.0020654, -0.0104521, 0.0016457, -0.0113204, 0.0125175
8: -0.0162758, 0.0105179, -0.0156893, 0.0130568, -0.0291874, 0.0260394
9: -0.0095560, 0.0092112, -0.0085069, 0.0088671, -0.0184231, 0.0177180

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 137

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0170924, upper bound: 0.0174393
time: 2.92 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0170924, upper bound: 0.0174540
time: 2.93 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0087994, 0.0077907, -0.0090036, 0.0080580, -0.0168574, 0.0167943
1: 0.9905690, 1.0144873, 0.9906914, 1.0148927, -0.0243237, 0.0237959
2: -0.0111232, 0.0079751, -0.0109103, 0.0081865, -0.0192304, 0.0188853
3: -0.0007490, 0.0052555, -0.0008062, 0.0052047, -0.0059537, 0.0060616
4: -0.0097276, 0.0072081, -0.0099964, 0.0070399, -0.0167675, 0.0172045
5: -0.0032280, 0.0117325, -0.0033210, 0.0120688, -0.0152967, 0.0150534
6: -0.0133918, 0.0023021, -0.0138069, 0.0023350, -0.0157268, 0.0161091
7: -0.0103937, 0.0016294, -0.0103081, 0.0017775, -0.0121712, 0.0119375
8: -0.0156664, 0.0128661, -0.0158735, 0.0125864, -0.0280730, 0.0285352
9: -0.0084661, 0.0088537, -0.0088364, 0.0089751, -0.0174412, 0.0176900

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 137

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0174786, upper bound: 0.0171312
time: 3.34 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0174786, upper bound: 0.0171597
time: 2.50 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0088460, 0.0078517, -0.0088219, 0.0078202, -0.0166661, 0.0166735
1: 0.9904509, 1.0145799, 0.9904856, 1.0145320, -0.0240811, 0.0240943
2: -0.0113285, 0.0080233, -0.0112683, 0.0079984, -0.0191819, 0.0192076
3: -0.0007621, 0.0053044, -0.0007553, 0.0052901, -0.0060521, 0.0060597
4: -0.0097889, 0.0073704, -0.0097572, 0.0073229, -0.0171118, 0.0171277
5: -0.0032492, 0.0118092, -0.0032382, 0.0117696, -0.0150187, 0.0150474
6: -0.0134865, 0.0023096, -0.0134376, 0.0023058, -0.0157922, 0.0157472
7: -0.0104763, 0.0016632, -0.0104521, 0.0016457, -0.0121221, 0.0121153
8: -0.0157137, 0.0131359, -0.0156893, 0.0130568, -0.0285839, 0.0286181
9: -0.0085505, 0.0088814, -0.0085069, 0.0088671, -0.0174176, 0.0173883

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 137

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0174887, upper bound: 0.0171737
time: 3.17 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0174887, upper bound: 0.0172002
time: 2.50 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 7.48 seconds
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.48
Output dim: 1, lower bound: -0.0169733, upper bound: 0.0173699
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.48
Output dim: 1, lower bound: -0.0169733, upper bound: 0.0173846
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.48
Output dim: 1, lower bound: -0.0169816, upper bound: 0.0174393
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.48
Output dim: 1, lower bound: -0.0169816, upper bound: 0.0174540
NS_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 7.48
Output dim: 1, lower bound: -0.0172862, upper bound: 0.0171312
NS_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 7.48
Output dim: 1, lower bound: -0.0172862, upper bound: 0.0171597
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.48
Output dim: 1, lower bound: -0.0172970, upper bound: 0.0171737
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.48
Output dim: 1, lower bound: -0.0172970, upper bound: 0.0172002
NS_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 7.48
Output dim: 1, lower bound: -0.0170465, upper bound: 0.0170843
NS_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 7.48
Output dim: 1, lower bound: -0.0170465, upper bound: 0.0170843
NS_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 7.48
Output dim: 1, lower bound: -0.0170654, upper bound: 0.0170843
NS_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 7.48
Output dim: 1, lower bound: -0.0170654, upper bound: 0.0170843
NS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 7.48
Output dim: 1, lower bound: -0.0170848, upper bound: 0.0170924
NS_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 7.48
Output dim: 1, lower bound: -0.0170848, upper bound: 0.0170924
NS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 7.48
Output dim: 1, lower bound: -0.0171000, upper bound: 0.0170924
NS_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 7.48
Output dim: 1, lower bound: -0.0171000, upper bound: 0.0170924
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.48
Output dim: 1, lower bound: -0.0170843, upper bound: 0.0173699
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.48
Output dim: 1, lower bound: -0.0170843, upper bound: 0.0173846
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.48
Output dim: 1, lower bound: -0.0170924, upper bound: 0.0174393
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.48
Output dim: 1, lower bound: -0.0170924, upper bound: 0.0174540
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.48
Output dim: 1, lower bound: -0.0174786, upper bound: 0.0171312
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.48
Output dim: 1, lower bound: -0.0174786, upper bound: 0.0171597
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.48
Output dim: 1, lower bound: -0.0174887, upper bound: 0.0171737
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.48
Output dim: 1, lower bound: -0.0174887, upper bound: 0.0172002

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0083967, 0.0072636, -0.0087956, 0.0077858, -0.0161825, 0.0160592
1: 0.9920977, 1.0136878, 0.9908590, 1.0144800, -0.0223823, 0.0228288
2: -0.0084640, 0.0075581, -0.0106185, 0.0079712, -0.0164352, 0.0181765
3: -0.0006363, 0.0046214, -0.0007480, 0.0051351, -0.0057714, 0.0053694
4: -0.0091976, 0.0051065, -0.0097227, 0.0068093, -0.0160069, 0.0148292
5: -0.0030446, 0.0110694, -0.0032262, 0.0117263, -0.0147709, 0.0142956
6: -0.0125732, 0.0022374, -0.0133841, 0.0023015, -0.0148747, 0.0156215
7: -0.0093239, 0.0013373, -0.0101907, 0.0016267, -0.0109506, 0.0115280
8: -0.0152582, 0.0093723, -0.0156626, 0.0122030, -0.0273220, 0.0248939
9: -0.0077358, 0.0086142, -0.0084592, 0.0088515, -0.0165873, 0.0170734

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 137

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0167354, upper bound: 0.0170832
time: 2.87 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0168096, upper bound: 0.0171953
time: 3.11 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0084696, 0.0073589, -0.0086994, 0.0076598, -0.0161294, 0.0160583
1: 0.9918648, 1.0138324, 0.9908642, 1.0142887, -0.0224239, 0.0229682
2: -0.0088690, 0.0076335, -0.0106099, 0.0078715, -0.0167405, 0.0182434
3: -0.0006567, 0.0047180, -0.0007210, 0.0051331, -0.0057898, 0.0054390
4: -0.0092935, 0.0054266, -0.0095960, 0.0068025, -0.0160960, 0.0150227
5: -0.0030777, 0.0111893, -0.0031824, 0.0115678, -0.0146456, 0.0143717
6: -0.0127212, 0.0022491, -0.0131885, 0.0022861, -0.0150073, 0.0154376
7: -0.0094868, 0.0013901, -0.0101872, 0.0015569, -0.0110437, 0.0115774
8: -0.0153321, 0.0099045, -0.0155651, 0.0121917, -0.0273895, 0.0253330
9: -0.0078679, 0.0086575, -0.0082847, 0.0087942, -0.0166622, 0.0169422

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 137

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0167354, upper bound: 0.0170933
time: 3.05 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0168096, upper bound: 0.0172085
time: 3.08 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0084400, 0.0073203, -0.0086177, 0.0075528, -0.0159929, 0.0159379
1: 0.9919907, 1.0137738, 0.9906530, 1.0141265, -0.0221359, 0.0231208
2: -0.0086502, 0.0076029, -0.0109770, 0.0077869, -0.0164370, 0.0185782
3: -0.0006484, 0.0046658, -0.0006982, 0.0052206, -0.0058690, 0.0053639
4: -0.0092546, 0.0052536, -0.0094885, 0.0070927, -0.0163473, 0.0147421
5: -0.0030643, 0.0111407, -0.0031452, 0.0114333, -0.0144976, 0.0142859
6: -0.0126612, 0.0022443, -0.0130224, 0.0022729, -0.0149341, 0.0152667
7: -0.0093988, 0.0013687, -0.0103350, 0.0014976, -0.0108964, 0.0117037
8: -0.0153021, 0.0096169, -0.0154822, 0.0126741, -0.0278327, 0.0249580
9: -0.0078144, 0.0086400, -0.0081366, 0.0087456, -0.0165600, 0.0167765

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 137

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0167402, upper bound: 0.0171432
time: 3.30 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0168168, upper bound: 0.0172602
time: 3.16 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0085130, 0.0074158, -0.0085087, 0.0074102, -0.0159232, 0.0159245
1: 0.9917607, 1.0139188, 0.9906577, 1.0139102, -0.0221495, 0.0232610
2: -0.0090504, 0.0076785, -0.0109687, 0.0076740, -0.0167244, 0.0185887
3: -0.0006688, 0.0047612, -0.0006676, 0.0052186, -0.0058875, 0.0054289
4: -0.0093507, 0.0055700, -0.0093450, 0.0070861, -0.0164368, 0.0149150
5: -0.0030975, 0.0112609, -0.0030955, 0.0112538, -0.0143513, 0.0143564
6: -0.0128095, 0.0022561, -0.0128008, 0.0022554, -0.0150649, 0.0150569
7: -0.0095598, 0.0014216, -0.0103316, 0.0014185, -0.0109784, 0.0117532
8: -0.0153761, 0.0101428, -0.0153718, 0.0126632, -0.0278994, 0.0253776
9: -0.0079467, 0.0086834, -0.0079389, 0.0086808, -0.0166275, 0.0166223

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 137

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0167402, upper bound: 0.0171527
time: 2.64 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0168168, upper bound: 0.0172739
time: 2.71 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0078733, 0.0065784, -0.0086177, 0.0075528, -0.0154261, 0.0151960
1: 0.9908487, 1.0126486, 0.9906530, 1.0141265, -0.0232779, 0.0219955
2: -0.0106367, 0.0070160, -0.0109770, 0.0077869, -0.0179852, 0.0179029
3: -0.0004898, 0.0051395, -0.0006982, 0.0052206, -0.0057104, 0.0058376
4: -0.0085087, 0.0068237, -0.0094885, 0.0070927, -0.0156014, 0.0163121
5: -0.0028061, 0.0102074, -0.0031452, 0.0114333, -0.0142394, 0.0133526
6: -0.0115090, 0.0021531, -0.0130224, 0.0022729, -0.0137819, 0.0151755
7: -0.0101980, 0.0009576, -0.0103350, 0.0014976, -0.0116956, 0.0112926
8: -0.0147276, 0.0122269, -0.0154822, 0.0126741, -0.0272124, 0.0275295
9: -0.0073054, 0.0083029, -0.0081366, 0.0087456, -0.0160510, 0.0164395

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 137

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0170501, upper bound: 0.0169232
time: 3.35 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0171330, upper bound: 0.0170187
time: 3.06 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0079498, 0.0066784, -0.0085087, 0.0074102, -0.0153599, 0.0151871
1: 0.9906305, 1.0128002, 0.9906577, 1.0139102, -0.0232797, 0.0221425
2: -0.0110163, 0.0070951, -0.0109687, 0.0076740, -0.0182448, 0.0179137
3: -0.0005112, 0.0052300, -0.0006676, 0.0052186, -0.0057298, 0.0058976
4: -0.0086093, 0.0071237, -0.0093450, 0.0070861, -0.0156954, 0.0164687
5: -0.0028410, 0.0103333, -0.0030955, 0.0112538, -0.0140948, 0.0134288
6: -0.0116644, 0.0021654, -0.0128008, 0.0022554, -0.0139198, 0.0149663
7: -0.0103507, 0.0010131, -0.0103316, 0.0014185, -0.0117693, 0.0113447
8: -0.0148051, 0.0127256, -0.0153718, 0.0126632, -0.0272851, 0.0279247
9: -0.0075823, 0.0083484, -0.0079389, 0.0086808, -0.0162631, 0.0162873

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 137

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0170501, upper bound: 0.0169462
time: 2.92 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0171330, upper bound: 0.0170423
time: 3.10 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0087699, 0.0077521, -0.0088123, 0.0078077, -0.0165775, 0.0165645
1: 0.9917259, 1.0144289, 0.9906991, 1.0145133, -0.0227874, 0.0237298
2: -0.0091107, 0.0079445, -0.0108971, 0.0079885, -0.0170992, 0.0188417
3: -0.0007408, 0.0047756, -0.0007526, 0.0052016, -0.0059423, 0.0055282
4: -0.0096888, 0.0056176, -0.0097447, 0.0070295, -0.0167184, 0.0153623
5: -0.0032145, 0.0116839, -0.0032338, 0.0117538, -0.0149684, 0.0149178
6: -0.0133319, 0.0022974, -0.0134181, 0.0023042, -0.0156361, 0.0157155
7: -0.0095841, 0.0016080, -0.0103028, 0.0016388, -0.0112228, 0.0119108
8: -0.0156366, 0.0102220, -0.0156796, 0.0125691, -0.0280658, 0.0257351
9: -0.0084126, 0.0088362, -0.0084895, 0.0088614, -0.0172740, 0.0173257

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 137

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0168419, upper bound: 0.0170832
time: 3.06 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0169152, upper bound: 0.0171953
time: 2.99 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0088383, 0.0078416, -0.0087158, 0.0076814, -0.0165196, 0.0165574
1: 0.9914988, 1.0145646, 0.9907039, 1.0143213, -0.0228226, 0.0238608
2: -0.0095060, 0.0080153, -0.0108885, 0.0078886, -0.0173945, 0.0189039
3: -0.0007599, 0.0048699, -0.0007256, 0.0051995, -0.0059594, 0.0055955
4: -0.0097788, 0.0059300, -0.0096177, 0.0070227, -0.0168015, 0.0155477
5: -0.0032456, 0.0117965, -0.0031899, 0.0115950, -0.0148406, 0.0149864
6: -0.0134708, 0.0023084, -0.0132220, 0.0022887, -0.0157595, 0.0155304
7: -0.0097431, 0.0016576, -0.0102993, 0.0015688, -0.0113119, 0.0119569
8: -0.0157058, 0.0107413, -0.0155818, 0.0125578, -0.0281274, 0.0261611
9: -0.0085366, 0.0088768, -0.0083146, 0.0088040, -0.0173406, 0.0171914

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 137

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0168419, upper bound: 0.0170933
time: 2.71 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0169152, upper bound: 0.0172085
time: 3.54 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0088128, 0.0078082, -0.0086351, 0.0075756, -0.0163884, 0.0164433
1: 0.9916205, 1.0145140, 0.9904930, 1.0141609, -0.0225404, 0.0240210
2: -0.0092939, 0.0079889, -0.0112553, 0.0078049, -0.0170988, 0.0191314
3: -0.0007528, 0.0048193, -0.0007030, 0.0052870, -0.0060397, 0.0055223
4: -0.0097452, 0.0057624, -0.0095114, 0.0073126, -0.0170578, 0.0152738
5: -0.0032340, 0.0117545, -0.0031531, 0.0114619, -0.0146959, 0.0149076
6: -0.0134190, 0.0023043, -0.0130577, 0.0022757, -0.0156947, 0.0153620
7: -0.0096578, 0.0016391, -0.0104469, 0.0015102, -0.0111680, 0.0120860
8: -0.0156800, 0.0104627, -0.0154999, 0.0130396, -0.0285748, 0.0257940
9: -0.0084903, 0.0088616, -0.0081681, 0.0087560, -0.0172463, 0.0170297

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 137

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0168517, upper bound: 0.0171432
time: 3.27 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0169227, upper bound: 0.0172602
time: 3.19 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0088820, 0.0078989, -0.0085259, 0.0074327, -0.0163147, 0.0164248
1: 0.9913916, 1.0146514, 0.9904979, 1.0139443, -0.0225527, 0.0241535
2: -0.0096924, 0.0080606, -0.0112469, 0.0076918, -0.0173842, 0.0191352
3: -0.0007722, 0.0049143, -0.0006725, 0.0052850, -0.0060571, 0.0055868
4: -0.0098364, 0.0060773, -0.0093677, 0.0073059, -0.0171423, 0.0154450
5: -0.0032656, 0.0118686, -0.0031034, 0.0112821, -0.0145478, 0.0149720
6: -0.0135598, 0.0023154, -0.0128358, 0.0022581, -0.0158180, 0.0151512
7: -0.0098181, 0.0016894, -0.0104435, 0.0014310, -0.0112491, 0.0121329
8: -0.0157502, 0.0109862, -0.0153892, 0.0130286, -0.0286364, 0.0262148
9: -0.0086159, 0.0089028, -0.0079701, 0.0086910, -0.0173070, 0.0168730

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 137

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0168517, upper bound: 0.0171527
time: 2.95 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0169226, upper bound: 0.0172739
time: 4.45 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0082079, 0.0070164, -0.0088123, 0.0078077, -0.0160156, 0.0158287
1: 0.9905930, 1.0133128, 0.9906991, 1.0145133, -0.0239202, 0.0226138
2: -0.0110813, 0.0073625, -0.0108971, 0.0079885, -0.0185854, 0.0182596
3: -0.0005834, 0.0052455, -0.0007526, 0.0052016, -0.0057850, 0.0059981
4: -0.0089491, 0.0071751, -0.0097447, 0.0070295, -0.0159787, 0.0169197
5: -0.0029586, 0.0107585, -0.0032338, 0.0117538, -0.0147124, 0.0139923
6: -0.0121893, 0.0022070, -0.0134181, 0.0023042, -0.0144935, 0.0156251
7: -0.0103769, 0.0012004, -0.0103028, 0.0016388, -0.0120157, 0.0115032
8: -0.0150668, 0.0128111, -0.0156796, 0.0125691, -0.0274552, 0.0282860
9: -0.0076298, 0.0085019, -0.0084895, 0.0088614, -0.0164912, 0.0169915

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 137

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0172319, upper bound: 0.0168929
time: 3.32 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0173063, upper bound: 0.0169806
time: 2.55 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0082991, 0.0071358, -0.0087158, 0.0076814, -0.0159805, 0.0158516
1: 0.9903654, 1.0134940, 0.9907039, 1.0143213, -0.0239559, 0.0227901
2: -0.0114775, 0.0074569, -0.0108885, 0.0078886, -0.0188428, 0.0183455
3: -0.0006090, 0.0053400, -0.0007256, 0.0051995, -0.0058085, 0.0060656
4: -0.0090692, 0.0074882, -0.0096177, 0.0070227, -0.0160919, 0.0171059
5: -0.0030001, 0.0109086, -0.0031899, 0.0115950, -0.0145951, 0.0140986
6: -0.0123747, 0.0022216, -0.0132220, 0.0022887, -0.0146634, 0.0154436
7: -0.0105363, 0.0012665, -0.0102993, 0.0015688, -0.0121051, 0.0115658
8: -0.0151593, 0.0133317, -0.0155818, 0.0125578, -0.0275393, 0.0287114
9: -0.0079189, 0.0085562, -0.0083146, 0.0088040, -0.0167229, 0.0168707

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 137

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0172319, upper bound: 0.0169172
time: 2.38 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0173063, upper bound: 0.0170035
time: 2.36 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0082543, 0.0070772, -0.0086351, 0.0075756, -0.0158299, 0.0157122
1: 0.9904751, 1.0134052, 0.9904930, 1.0141609, -0.0236858, 0.0229122
2: -0.0112865, 0.0074106, -0.0112553, 0.0078049, -0.0185547, 0.0184488
3: -0.0005964, 0.0052944, -0.0007030, 0.0052870, -0.0058834, 0.0059974
4: -0.0090102, 0.0073372, -0.0095114, 0.0073126, -0.0163228, 0.0168486
5: -0.0029797, 0.0108349, -0.0031531, 0.0114619, -0.0144416, 0.0139880
6: -0.0122837, 0.0022144, -0.0130577, 0.0022757, -0.0145594, 0.0152722
7: -0.0104594, 0.0012340, -0.0104469, 0.0015102, -0.0119696, 0.0116809
8: -0.0151139, 0.0130807, -0.0154999, 0.0130396, -0.0279655, 0.0283727
9: -0.0077795, 0.0085295, -0.0081681, 0.0087560, -0.0165354, 0.0166976

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 137

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0172380, upper bound: 0.0169232
time: 2.88 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0173168, upper bound: 0.0170187
time: 3.74 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0083467, 0.0071981, -0.0085259, 0.0074327, -0.0157794, 0.0157241
1: 0.9902477, 1.0135887, 0.9904979, 1.0139443, -0.0236966, 0.0230908
2: -0.0116820, 0.0075063, -0.0112469, 0.0076918, -0.0188311, 0.0184756
3: -0.0006223, 0.0053887, -0.0006725, 0.0052850, -0.0059073, 0.0060612
4: -0.0091318, 0.0076498, -0.0093677, 0.0073059, -0.0164378, 0.0170175
5: -0.0030218, 0.0109871, -0.0031034, 0.0112821, -0.0143039, 0.0140905
6: -0.0124715, 0.0022293, -0.0128358, 0.0022581, -0.0147297, 0.0150651
7: -0.0106186, 0.0013010, -0.0104435, 0.0014310, -0.0120496, 0.0117445
8: -0.0152075, 0.0136003, -0.0153892, 0.0130286, -0.0280518, 0.0287875
9: -0.0080681, 0.0085845, -0.0079701, 0.0086910, -0.0167591, 0.0165546

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 137

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0172380, upper bound: 0.0169462
time: 3.23 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0173168, upper bound: 0.0170422
time: 3.01 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 7.84 seconds
NS_A1_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 7.84
Output dim: 1, lower bound: -0.0167354, upper bound: 0.0170832
NS_A1_B2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 7.84
Output dim: 1, lower bound: -0.0168096, upper bound: 0.0171953
NS_A1_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 7.84
Output dim: 1, lower bound: -0.0167354, upper bound: 0.0170933
NS_A1_B2_A1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 7.84
Output dim: 1, lower bound: -0.0168096, upper bound: 0.0172085
NS_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 7.84
Output dim: 1, lower bound: -0.0167402, upper bound: 0.0171432
NS_A1_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 7.84
Output dim: 1, lower bound: -0.0168168, upper bound: 0.0172602
NS_A1_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 7.84
Output dim: 1, lower bound: -0.0167402, upper bound: 0.0171527
NS_A1_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 7.84
Output dim: 1, lower bound: -0.0168168, upper bound: 0.0172739
NS_A1_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 7.84
Output dim: 1, lower bound: -0.0170501, upper bound: 0.0169232
NS_A1_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 7.84
Output dim: 1, lower bound: -0.0171330, upper bound: 0.0170187
NS_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 7.84
Output dim: 1, lower bound: -0.0170501, upper bound: 0.0169462
NS_A1_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 7.84
Output dim: 1, lower bound: -0.0171330, upper bound: 0.0170423
NS_A2_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 7.84
Output dim: 1, lower bound: -0.0168419, upper bound: 0.0170832
NS_A2_B2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 7.84
Output dim: 1, lower bound: -0.0169152, upper bound: 0.0171953
NS_A2_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 7.84
Output dim: 1, lower bound: -0.0168419, upper bound: 0.0170933
NS_A2_B2_A1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 7.84
Output dim: 1, lower bound: -0.0169152, upper bound: 0.0172085
NS_A2_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 7.84
Output dim: 1, lower bound: -0.0168517, upper bound: 0.0171432
NS_A2_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 7.84
Output dim: 1, lower bound: -0.0169227, upper bound: 0.0172602
NS_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 7.84
Output dim: 1, lower bound: -0.0168517, upper bound: 0.0171527
NS_A2_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 7.84
Output dim: 1, lower bound: -0.0169226, upper bound: 0.0172739
NS_A2_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 7.84
Output dim: 1, lower bound: -0.0172319, upper bound: 0.0168929
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.84
Output dim: 1, lower bound: -0.0173063, upper bound: 0.0169806
NS_A2_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 7.84
Output dim: 1, lower bound: -0.0172319, upper bound: 0.0169172
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.84
Output dim: 1, lower bound: -0.0173063, upper bound: 0.0170035
NS_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 7.84
Output dim: 1, lower bound: -0.0172380, upper bound: 0.0169232
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.84
Output dim: 1, lower bound: -0.0173168, upper bound: 0.0170187
NS_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 7.84
Output dim: 1, lower bound: -0.0172380, upper bound: 0.0169462
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.84
Output dim: 1, lower bound: -0.0173168, upper bound: 0.0170422

## BFS NS instance: NS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0081142, 0.0068938, -0.0084327, 0.0073107, -0.0154249, 0.0153265
1: 0.9905964, 1.0131269, 0.9907135, 1.0137593, -0.0231628, 0.0224134
2: -0.0110753, 0.0072655, -0.0108719, 0.0075953, -0.0181445, 0.0179452
3: -0.0005572, 0.0052441, -0.0006464, 0.0051955, -0.0057527, 0.0058904
4: -0.0088258, 0.0071703, -0.0092450, 0.0070096, -0.0158354, 0.0164153
5: -0.0029159, 0.0106042, -0.0030609, 0.0111287, -0.0140445, 0.0136651
6: -0.0119988, 0.0021919, -0.0126463, 0.0022431, -0.0142419, 0.0148382
7: -0.0103745, 0.0011324, -0.0102927, 0.0013634, -0.0117379, 0.0114250
8: -0.0149718, 0.0128032, -0.0152947, 0.0125359, -0.0273270, 0.0278948
9: -0.0076254, 0.0084462, -0.0078011, 0.0086356, -0.0162610, 0.0162473

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 137

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0172023, upper bound: 0.0168967
time: 3.05 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0172176, upper bound: 0.0168967
time: 2.69 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0081954, 0.0070001, -0.0083287, 0.0071746, -0.0153700, 0.0153288
1: 0.9903689, 1.0132879, 0.9907183, 1.0135528, -0.0231839, 0.0225696
2: -0.0114714, 0.0073495, -0.0108634, 0.0074876, -0.0184342, 0.0179608
3: -0.0005799, 0.0053385, -0.0006173, 0.0051935, -0.0057735, 0.0059557
4: -0.0089327, 0.0074833, -0.0091081, 0.0070028, -0.0159355, 0.0165915
5: -0.0029529, 0.0107379, -0.0030136, 0.0109574, -0.0139103, 0.0137514
6: -0.0121639, 0.0022050, -0.0124349, 0.0022264, -0.0143903, 0.0146399
7: -0.0105338, 0.0011913, -0.0102892, 0.0012880, -0.0118218, 0.0114805
8: -0.0150541, 0.0133236, -0.0151893, 0.0125248, -0.0274011, 0.0283149
9: -0.0079144, 0.0084945, -0.0076125, 0.0085738, -0.0164881, 0.0161070

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 137

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0172023, upper bound: 0.0169214
time: 3.15 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0172176, upper bound: 0.0169214
time: 3.03 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0081607, 0.0069547, -0.0082578, 0.0070817, -0.0152425, 0.0152125
1: 0.9904786, 1.0132192, 0.9905083, 1.0134119, -0.0229333, 0.0227110
2: -0.0112804, 0.0073137, -0.0112291, 0.0074142, -0.0181504, 0.0180928
3: -0.0005702, 0.0052930, -0.0005974, 0.0052807, -0.0058509, 0.0058904
4: -0.0088871, 0.0073324, -0.0090148, 0.0072918, -0.0161789, 0.0163473
5: -0.0029371, 0.0106808, -0.0029813, 0.0108407, -0.0137778, 0.0136621
6: -0.0120934, 0.0021994, -0.0122908, 0.0022150, -0.0143084, 0.0144901
7: -0.0104570, 0.0011661, -0.0104364, 0.0012366, -0.0116936, 0.0116025
8: -0.0150190, 0.0130727, -0.0151174, 0.0130052, -0.0278361, 0.0279848
9: -0.0077751, 0.0084739, -0.0077376, 0.0085316, -0.0163067, 0.0162114

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 137

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0172125, upper bound: 0.0169327
time: 3.15 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0172280, upper bound: 0.0169327
time: 3.41 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0082431, 0.0070625, -0.0081477, 0.0069375, -0.0151806, 0.0152101
1: 0.9902514, 1.0133827, 0.9905130, 1.0131932, -0.0229418, 0.0228697
2: -0.0116758, 0.0073989, -0.0112207, 0.0073001, -0.0184296, 0.0181492
3: -0.0005933, 0.0053872, -0.0005666, 0.0052787, -0.0058720, 0.0059538
4: -0.0089955, 0.0076449, -0.0088698, 0.0072853, -0.0162807, 0.0165148
5: -0.0029746, 0.0108164, -0.0029311, 0.0106592, -0.0136338, 0.0137475
6: -0.0122608, 0.0022126, -0.0120668, 0.0021973, -0.0144581, 0.0142794
7: -0.0106161, 0.0012259, -0.0104330, 0.0011567, -0.0117727, 0.0116589
8: -0.0151025, 0.0135922, -0.0150057, 0.0129943, -0.0279124, 0.0283978
9: -0.0080636, 0.0085228, -0.0077315, 0.0084661, -0.0165296, 0.0162544

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 137

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0172125, upper bound: 0.0169577
time: 2.90 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0172280, upper bound: 0.0169577
time: 3.10 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 7.80 seconds
NS_A2_B2_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 7.80
Output dim: 1, lower bound: -0.0172023, upper bound: 0.0168967
NS_A2_B2_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 7.80
Output dim: 1, lower bound: -0.0172176, upper bound: 0.0168967
NS_A2_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 7.80
Output dim: 1, lower bound: -0.0172023, upper bound: 0.0169214
NS_A2_B2_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 7.80
Output dim: 1, lower bound: -0.0172176, upper bound: 0.0169214
NS_A2_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 7.80
Output dim: 1, lower bound: -0.0172125, upper bound: 0.0169327
NS_A2_B2_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 7.80
Output dim: 1, lower bound: -0.0172280, upper bound: 0.0169327
NS_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 7.80
Output dim: 1, lower bound: -0.0172125, upper bound: 0.0169577
NS_A2_B2_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 7.80
Output dim: 1, lower bound: -0.0172280, upper bound: 0.0169577

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 5.06 + 348.68 = 353.74 seconds
