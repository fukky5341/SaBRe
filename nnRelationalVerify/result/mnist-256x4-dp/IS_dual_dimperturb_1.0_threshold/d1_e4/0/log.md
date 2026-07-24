## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.037597364


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0949643, 0.0846384, -0.0949643, 0.0846384, -0.1796028, 0.1796028)
1: (-0.0607208, 0.0474269, -0.0607208, 0.0474269, -0.1081477, 0.1081477)
2: (-0.1246243, 0.0634956, -0.1246243, 0.0634956, -0.1881199, 0.1881199)
3: (0.9865127, 1.0434601, 0.9865127, 1.0434601, -0.0569474, 0.0569474)
4: (-0.0387707, 0.1049149, -0.0387707, 0.1049149, -0.1436856, 0.1436856)
5: (-0.0581584, 0.1370523, -0.0581584, 0.1370523, -0.1952107, 0.1952107)
6: (-0.1140255, 0.0998719, -0.1140255, 0.0998719, -0.2138974, 0.2138974)
7: (-0.0948062, 0.0074982, -0.0948062, 0.0074982, -0.1023044, 0.1023044)
8: (-0.0549271, 0.0979662, -0.0549271, 0.0979662, -0.1528933, 0.1528933)
9: (-0.0670059, 0.0810942, -0.0670059, 0.0810942, -0.1481002, 0.1481002)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.26 + 3.11 = 4.37 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.0408667, upper bound: 0.0408667

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0402266, upper bound: 0.0396385
time: 3.76 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0405998, upper bound: 0.0405998
time: 2.12 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 6.01 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 6.01
Output dim: 3, lower bound: -0.0402266, upper bound: 0.0396385
IS_B2, status: Status.UNKNOWN, split count: 1, time: 6.01
Output dim: 3, lower bound: -0.0405998, upper bound: 0.0405998

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -0.0644543, 0.0542954, -0.0292444, 0.0290629, -0.0935172, 0.0835398
1: -0.0435974, 0.0365575, -0.0216143, 0.0239352, -0.0675326, 0.0581718
2: -0.0936717, 0.0379863, -0.0545891, 0.0171072, -0.1107789, 0.0925754
3: 0.9954457, 1.0362935, 1.0031919, 1.0266957, -0.0312501, 0.0331016
4: -0.0237957, 0.0714007, -0.0091080, 0.0363618, -0.0601575, 0.0805088
5: -0.0345231, 0.1088749, -0.0118294, 0.0729811, -0.1075041, 0.1207043
6: -0.0866648, 0.0625102, -0.0551442, 0.0220338, -0.1086985, 0.1176545
7: -0.0711236, -0.0000413, -0.0459556, -0.0018354, -0.0692351, 0.0459142
8: -0.0417600, 0.0588747, -0.0255307, 0.0210990, -0.0628590, 0.0844054
9: -0.0452756, 0.0622656, -0.0250729, 0.0401239, -0.0853995, 0.0873386

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=7, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_B1_B1

### Relational analysis result of IS_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0396971, upper bound: 0.0390692
time: 2.28 seconds

## Relational analysis of IS_B1_B2

### Relational analysis result of IS_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0396971, upper bound: 0.0393134
time: 2.23 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -0.0870512, 0.0767454, -0.0730199, 0.0627461, -0.1497973, 0.1497653
1: -0.0562825, 0.0445735, -0.0484077, 0.0395258, -0.0958083, 0.0929813
2: -0.1165480, 0.0569014, -0.1022281, 0.0452114, -0.1617594, 0.1591295
3: 0.9888328, 1.0415614, 0.9929404, 1.0382117, -0.0493789, 0.0486210
4: -0.0348982, 0.0961992, -0.0280313, 0.0807345, -0.1156327, 0.1242305
5: -0.0520034, 0.1297383, -0.0411120, 0.1167367, -0.1687401, 0.1708503
6: -0.1068997, 0.0901646, -0.0942560, 0.0729583, -0.1798580, 0.1844205
7: -0.0886601, 0.0053062, -0.0777540, 0.0017832, -0.0904433, 0.0830602
8: -0.0514881, 0.0878114, -0.0454056, 0.0698065, -0.1212946, 0.1332170
9: -0.0613672, 0.0761822, -0.0513704, 0.0674620, -0.1288292, 0.1275526

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_B2_A1

### Relational analysis result of IS_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0400269, upper bound: 0.0401918
time: 3.05 seconds

## Relational analysis of IS_B2_A2

### Relational analysis result of IS_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0402792, upper bound: 0.0402792
time: 2.53 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 6.86 seconds
IS_B1_B1, status: Status.UNKNOWN, split count: 2, time: 6.86
Output dim: 3, lower bound: -0.0396971, upper bound: 0.0390692
IS_B1_B2, status: Status.UNKNOWN, split count: 2, time: 6.86
Output dim: 3, lower bound: -0.0396971, upper bound: 0.0393134
IS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 6.86
Output dim: 3, lower bound: -0.0400269, upper bound: 0.0401918
IS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 6.86
Output dim: 3, lower bound: -0.0402792, upper bound: 0.0402792

## BFS IS instance: IS_B1_B1

### Backsubstitution after applying IS history:
0: -0.0448966, 0.0352413, -0.0098695, 0.0191215, -0.0640180, 0.0451108
1: -0.0326266, 0.0292777, -0.0096381, 0.0119354, -0.0445620, 0.0389158
2: -0.0732528, 0.0221941, -0.0208713, 0.0165554, -0.0898082, 0.0430654
3: 1.0007942, 1.0312525, 1.0050359, 1.0162131, -0.0154189, 0.0262166
4: -0.0145367, 0.0497584, -0.0042441, 0.0145553, -0.0290921, 0.0540026
5: -0.0193217, 0.0905951, -0.0018259, 0.0431341, -0.0624558, 0.0924210
6: -0.0687938, 0.0386090, -0.0285090, 0.0044036, -0.0731975, 0.0671180
7: -0.0558972, -0.0014705, -0.0310072, -0.0014708, -0.0543696, 0.0295367
8: -0.0330296, 0.0342666, -0.0147542, 0.0063967, -0.0394263, 0.0490209
9: -0.0315475, 0.0499918, -0.0154371, 0.0194150, -0.0509625, 0.0654289

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 207

## Relational analysis of IS_B1_B1_B1

### Relational analysis result of IS_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0391533, upper bound: 0.0387793
time: 2.03 seconds

## Relational analysis of IS_B1_B1_B2

### Relational analysis result of IS_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0390844, upper bound: 0.0383463
time: 2.61 seconds

## BFS IS instance: IS_B1_B2

### Backsubstitution after applying IS history:
0: -0.0571760, 0.0469512, -0.0160453, 0.0234805, -0.0806564, 0.0629966
1: -0.0395152, 0.0337707, -0.0128137, 0.0171739, -0.0566891, 0.0465845
2: -0.0859663, 0.0320881, -0.0340622, 0.0167971, -0.1027634, 0.0661503
3: 0.9975322, 1.0344002, 1.0042502, 1.0203801, -0.0228479, 0.0301501
4: -0.0203285, 0.0632231, -0.0042793, 0.0238921, -0.0442207, 0.0675024
5: -0.0288113, 0.1020315, -0.0048228, 0.0562625, -0.0850738, 0.1068542
6: -0.0799551, 0.0534869, -0.0396624, 0.0070156, -0.0869707, 0.0931493
7: -0.0654223, -0.0009903, -0.0371904, -0.0018536, -0.0635168, 0.0362001
8: -0.0385001, 0.0495163, -0.0190440, 0.0089871, -0.0474872, 0.0685604
9: -0.0401297, 0.0575652, -0.0191680, 0.0285083, -0.0686380, 0.0767332

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 207

## Relational analysis of IS_B1_B2_B1

### Relational analysis result of IS_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0394273, upper bound: 0.0390586
time: 4.39 seconds

## Relational analysis of IS_B1_B2_B2

### Relational analysis result of IS_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393914, upper bound: 0.0387005
time: 2.47 seconds

## BFS IS instance: IS_B2_A1

### Backsubstitution after applying IS history:
0: -0.0433574, 0.0345143, -0.0535022, 0.0431518, -0.0865092, 0.0880165
1: -0.0317714, 0.0284229, -0.0374622, 0.0321323, -0.0639036, 0.0658851
2: -0.0710679, 0.0216214, -0.0817053, 0.0293292, -0.1003971, 0.1033267
3: 1.0009723, 1.0303112, 0.9985441, 1.0331829, -0.0322106, 0.0317671
4: -0.0142191, 0.0480325, -0.0187039, 0.0589054, -0.0731245, 0.0667364
5: -0.0180124, 0.0889289, -0.0258316, 0.0984843, -0.1164967, 0.1147606
6: -0.0670113, 0.0369942, -0.0763871, 0.0487744, -0.1157857, 0.1133814
7: -0.0546605, -0.0011119, -0.0624963, -0.0012283, -0.0534322, 0.0613514
8: -0.0320133, 0.0330373, -0.0366963, 0.0447546, -0.0767678, 0.0697335
9: -0.0306828, 0.0489615, -0.0375856, 0.0549783, -0.0856611, 0.0865472

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of IS_B2_A1_B1

### Relational analysis result of IS_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0397558, upper bound: 0.0397565
time: 3.19 seconds

## Relational analysis of IS_B2_A1_B2

### Relational analysis result of IS_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0397595, upper bound: 0.0399469
time: 2.48 seconds

## BFS IS instance: IS_B2_A2

### Backsubstitution after applying IS history:
0: -0.0619023, 0.0514714, -0.0656720, 0.0553415, -0.1172438, 0.1171434
1: -0.0421786, 0.0350573, -0.0442900, 0.0367270, -0.0789056, 0.0793473
2: -0.0901260, 0.0364094, -0.0944863, 0.0392184, -0.1293444, 0.1308957
3: 0.9960878, 1.0350876, 0.9950597, 1.0363010, -0.0402132, 0.0400279
4: -0.0228552, 0.0680838, -0.0245115, 0.0725057, -0.0953609, 0.0925953
5: -0.0322895, 0.1062267, -0.0353261, 0.1098787, -0.1421681, 0.1415528
6: -0.0838649, 0.0590107, -0.0875133, 0.0638335, -0.1476985, 0.1465240
7: -0.0690047, -0.0004036, -0.0720064, 0.0001973, -0.0692021, 0.0716028
8: -0.0402702, 0.0554797, -0.0421092, 0.0603465, -0.1006167, 0.0975889
9: -0.0435686, 0.0601041, -0.0461650, 0.0627594, -0.1063280, 0.1062692

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of IS_B2_A2_A1

### Relational analysis result of IS_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0398501, upper bound: 0.0400317
time: 2.54 seconds

## Relational analysis of IS_B2_A2_A2

### Relational analysis result of IS_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0400401, upper bound: 0.0400401
time: 2.60 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 6.43 seconds
IS_B1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 6.43
Output dim: 3, lower bound: -0.0391533, upper bound: 0.0387793
IS_B1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 6.43
Output dim: 3, lower bound: -0.0390844, upper bound: 0.0383463
IS_B1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 6.43
Output dim: 3, lower bound: -0.0394273, upper bound: 0.0390586
IS_B1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 6.43
Output dim: 3, lower bound: -0.0393914, upper bound: 0.0387005
IS_B2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 6.43
Output dim: 3, lower bound: -0.0397558, upper bound: 0.0397565
IS_B2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 6.43
Output dim: 3, lower bound: -0.0397595, upper bound: 0.0399469
IS_B2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 6.43
Output dim: 3, lower bound: -0.0398501, upper bound: 0.0400317
IS_B2_A2_A2, status: Status.UNKNOWN, split count: 3, time: 6.43
Output dim: 3, lower bound: -0.0400401, upper bound: 0.0400401

## BFS IS instance: IS_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0442179, 0.0349950, -0.0090652, 0.0185490, -0.0627670, 0.0440603
1: -0.0322466, 0.0290732, -0.0092262, 0.0112320, -0.0434787, 0.0382994
2: -0.0726007, 0.0218415, -0.0191230, 0.0165207, -0.0891214, 0.0409645
3: 1.0009439, 1.0310693, 1.0051676, 1.0156380, -0.0146941, 0.0259017
4: -0.0143683, 0.0490988, -0.0042391, 0.0133310, -0.0276993, 0.0533379
5: -0.0189157, 0.0899836, -0.0015160, 0.0414344, -0.0603501, 0.0914996
6: -0.0683001, 0.0378889, -0.0270349, 0.0040445, -0.0723447, 0.0649238
7: -0.0553984, -0.0015862, -0.0301953, -0.0018843, -0.0534586, 0.0286091
8: -0.0327199, 0.0337342, -0.0141690, 0.0059297, -0.0386496, 0.0479033
9: -0.0312181, 0.0496645, -0.0149386, 0.0182239, -0.0494420, 0.0646030

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_B1_B1_B1_A1

### Relational analysis result of IS_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0390412, upper bound: 0.0386830
time: 2.11 seconds

## Relational analysis of IS_B1_B1_B1_A2

### Relational analysis result of IS_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0390384, upper bound: 0.0386836
time: 2.11 seconds

## BFS IS instance: IS_B1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0433746, 0.0346853, -0.0280353, 0.0319714, -0.0753460, 0.0627207
1: -0.0317748, 0.0288107, -0.0189245, 0.0276344, -0.0594092, 0.0477351
2: -0.0717756, 0.0214090, -0.0599535, 0.0172283, -0.0890039, 0.0813624
3: 1.0011249, 1.0308304, 1.0027272, 1.0289898, -0.0278649, 0.0281032
4: -0.0141613, 0.0482730, -0.0043396, 0.0420205, -0.0561818, 0.0526127
5: -0.0184035, 0.0892157, -0.0121101, 0.0813691, -0.0997726, 0.1013258
6: -0.0676719, 0.0369965, -0.0615042, 0.0123819, -0.0800538, 0.0985007
7: -0.0547764, -0.0016964, -0.0491815, -0.0018757, -0.0528449, 0.0474835
8: -0.0323285, 0.0330742, -0.0277046, 0.0151869, -0.0475154, 0.0607788
9: -0.0308064, 0.0492509, -0.0265146, 0.0460004, -0.0768068, 0.0757654

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B1_B1_B2_B1

### Relational analysis result of IS_B1_B1_B2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0369544, upper bound: 0.0357803
time: 2.28 seconds

## Relational analysis of IS_B1_B1_B2_B2

### Relational analysis result of IS_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0390329, upper bound: 0.0382878
time: 2.54 seconds

## BFS IS instance: IS_B1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0564807, 0.0462365, -0.0151913, 0.0228760, -0.0793567, 0.0614278
1: -0.0391262, 0.0335001, -0.0123779, 0.0164320, -0.0555582, 0.0458779
2: -0.0852254, 0.0315134, -0.0322188, 0.0167621, -0.1019875, 0.0637322
3: 0.9977335, 1.0342109, 1.0043912, 1.0197716, -0.0220380, 0.0298197
4: -0.0199897, 0.0624411, -0.0042742, 0.0226029, -0.0425926, 0.0667153
5: -0.0282521, 0.1013879, -0.0043081, 0.0544714, -0.0827234, 0.1056960
6: -0.0793096, 0.0526138, -0.0381095, 0.0066366, -0.0859462, 0.0907233
7: -0.0648770, -0.0011360, -0.0363355, -0.0022549, -0.0625727, 0.0351995
8: -0.0381812, 0.0486075, -0.0184224, 0.0085124, -0.0466936, 0.0670298
9: -0.0396311, 0.0571191, -0.0186421, 0.0272570, -0.0668881, 0.0757612

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_B1_B2_B1_B1

### Relational analysis result of IS_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392640, upper bound: 0.0389339
time: 3.25 seconds

## Relational analysis of IS_B1_B2_B1_B2

### Relational analysis result of IS_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392633, upper bound: 0.0389249
time: 2.22 seconds

## BFS IS instance: IS_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0556303, 0.0453619, -0.0342101, 0.0363242, -0.0919545, 0.0795719
1: -0.0386505, 0.0331578, -0.0220918, 0.0329241, -0.0715746, 0.0552497
2: -0.0843009, 0.0308265, -0.0731770, 0.0174654, -0.1017663, 0.1040035
3: 0.9979772, 1.0339690, 1.0019752, 1.0332438, -0.0352666, 0.0319939
4: -0.0195852, 0.0614745, -0.0043734, 0.0513327, -0.0709179, 0.0658479
5: -0.0275672, 0.1005926, -0.0157307, 0.0943764, -0.1219436, 0.1163233
6: -0.0785120, 0.0515404, -0.0726721, 0.0150469, -0.0935588, 0.1242126
7: -0.0642073, -0.0012898, -0.0553578, -0.0022629, -0.0618919, 0.0540680
8: -0.0377850, 0.0474988, -0.0321185, 0.0183548, -0.0561398, 0.0796173
9: -0.0390268, 0.0565604, -0.0302650, 0.0550448, -0.0940716, 0.0868254

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of IS_B1_B2_B2_B1

### Relational analysis result of IS_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0391096, upper bound: 0.0383476
time: 2.44 seconds

## Relational analysis of IS_B1_B2_B2_B2

### Relational analysis result of IS_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0391210, upper bound: 0.0383667
time: 4.24 seconds

## BFS IS instance: IS_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0361455, 0.0313542, -0.0319514, 0.0299586, -0.0661041, 0.0633057
1: -0.0267745, 0.0260218, -0.0235492, 0.0248690, -0.0516435, 0.0495710
2: -0.0628840, 0.0180037, -0.0581108, 0.0171799, -0.0800639, 0.0761144
3: 1.0022784, 1.0282758, 1.0029564, 1.0274388, -0.0251603, 0.0253195
4: -0.0120033, 0.0415948, -0.0103843, 0.0383902, -0.0503936, 0.0519791
5: -0.0140691, 0.0808766, -0.0127421, 0.0759508, -0.0900199, 0.0936188
6: -0.0608480, 0.0297988, -0.0575279, 0.0254861, -0.0863341, 0.0873268
7: -0.0496670, -0.0012510, -0.0474805, -0.0016851, -0.0479414, 0.0461806
8: -0.0282237, 0.0273844, -0.0264104, 0.0238835, -0.0521072, 0.0537948
9: -0.0275837, 0.0447373, -0.0260471, 0.0421592, -0.0697429, 0.0707844

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 207

## Relational analysis of IS_B2_A1_B1_B1

### Relational analysis result of IS_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392938, upper bound: 0.0394325
time: 2.10 seconds

## Relational analysis of IS_B2_A1_B1_B2

### Relational analysis result of IS_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392886, upper bound: 0.0392294
time: 2.24 seconds

## BFS IS instance: IS_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0366687, 0.0316113, -0.0373917, 0.0320427, -0.0687114, 0.0690029
1: -0.0271641, 0.0262220, -0.0276178, 0.0268248, -0.0539889, 0.0538398
2: -0.0635601, 0.0182398, -0.0649699, 0.0184387, -0.0819987, 0.0832097
3: 1.0021909, 1.0284345, 1.0021429, 1.0291682, -0.0269773, 0.0262916
4: -0.0121705, 0.0420646, -0.0123334, 0.0429221, -0.0550925, 0.0543980
5: -0.0143477, 0.0815420, -0.0151934, 0.0824237, -0.0967714, 0.0967354
6: -0.0613680, 0.0302951, -0.0625385, 0.0310977, -0.0924657, 0.0928336
7: -0.0500340, -0.0012481, -0.0506347, -0.0013184, -0.0487103, 0.0493378
8: -0.0285106, 0.0277924, -0.0292295, 0.0284135, -0.0569241, 0.0570219
9: -0.0278036, 0.0450946, -0.0282458, 0.0456896, -0.0734932, 0.0733404

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 229

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B2_A1_B2_A1

### Relational analysis result of IS_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0365662, upper bound: 0.0376343
time: 2.67 seconds

## Relational analysis of IS_B2_A1_B2_A2

### Relational analysis result of IS_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0397183, upper bound: 0.0399099
time: 2.85 seconds

## BFS IS instance: IS_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0379009, 0.0323111, -0.0565611, 0.0463215, -0.0842223, 0.0888722
1: -0.0279783, 0.0270702, -0.0391739, 0.0335705, -0.0615487, 0.0662441
2: -0.0657127, 0.0186259, -0.0853812, 0.0315215, -0.0972342, 0.1040071
3: 1.0021034, 1.0294116, 0.9977205, 1.0342573, -0.0321538, 0.0316911
4: -0.0124721, 0.0434293, -0.0199945, 0.0625723, -0.0750444, 0.0634237
5: -0.0155162, 0.0831194, -0.0283181, 0.1014783, -0.1169945, 0.1114375
6: -0.0631241, 0.0315606, -0.0794073, 0.0527538, -0.1158779, 0.1109679
7: -0.0510108, -0.0016212, -0.0649576, -0.0010565, -0.0499543, 0.0632936
8: -0.0295309, 0.0288042, -0.0382416, 0.0487192, -0.0782501, 0.0670459
9: -0.0284791, 0.0460846, -0.0396698, 0.0572103, -0.0856894, 0.0857545

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 108

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B2_A2_A1_B1

### Relational analysis result of IS_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0378292, upper bound: 0.0369289
time: 2.09 seconds

## Relational analysis of IS_B2_A2_A1_B2

### Relational analysis result of IS_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0398111, upper bound: 0.0399960
time: 2.64 seconds

## BFS IS instance: IS_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0443613, 0.0350062, -0.0576964, 0.0474204, -0.0917817, 0.0927025
1: -0.0323218, 0.0290493, -0.0398047, 0.0338648, -0.0661867, 0.0688540
2: -0.0726085, 0.0219660, -0.0863316, 0.0326250, -0.1052336, 0.1082976
3: 1.0008518, 1.0310365, 0.9973678, 1.0344294, -0.0335777, 0.0336687
4: -0.0144175, 0.0491963, -0.0206364, 0.0637137, -0.0781312, 0.0698327
5: -0.0189656, 0.0900178, -0.0291939, 0.1024230, -0.1213886, 0.1192117
6: -0.0682790, 0.0380678, -0.0803263, 0.0540941, -0.1223731, 0.1183941
7: -0.0554868, -0.0011732, -0.0658038, -0.0009628, -0.0545240, 0.0646109
8: -0.0327573, 0.0338560, -0.0386788, 0.0501757, -0.0829329, 0.0725348
9: -0.0312737, 0.0496607, -0.0405203, 0.0577689, -0.0890425, 0.0901810

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 1

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 207

## Relational analysis of IS_B2_A2_A2_A1

### Relational analysis result of IS_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0399235, upper bound: 0.0396998
time: 3.44 seconds

## Relational analysis of IS_B2_A2_A2_A2

### Relational analysis result of IS_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0396987, upper bound: 0.0396987
time: 2.63 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 7.35 seconds
IS_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 7.35
Output dim: 3, lower bound: -0.0390412, upper bound: 0.0386830
IS_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 7.35
Output dim: 3, lower bound: -0.0390384, upper bound: 0.0386836
IS_B1_B1_B2_B1, status: Status.VERIFIED, split count: 4, time: 7.35
Output dim: 3, lower bound: -0.0369544, upper bound: 0.0357803
IS_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 7.35
Output dim: 3, lower bound: -0.0390329, upper bound: 0.0382878
IS_B1_B2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 7.35
Output dim: 3, lower bound: -0.0392640, upper bound: 0.0389339
IS_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 7.35
Output dim: 3, lower bound: -0.0392633, upper bound: 0.0389249
IS_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 7.35
Output dim: 3, lower bound: -0.0391096, upper bound: 0.0383476
IS_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 7.35
Output dim: 3, lower bound: -0.0391210, upper bound: 0.0383667
IS_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 7.35
Output dim: 3, lower bound: -0.0392938, upper bound: 0.0394325
IS_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 7.35
Output dim: 3, lower bound: -0.0392886, upper bound: 0.0392294
IS_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 7.35
Output dim: 3, lower bound: -0.0365662, upper bound: 0.0376343
IS_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 7.35
Output dim: 3, lower bound: -0.0397183, upper bound: 0.0399099
IS_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 7.35
Output dim: 3, lower bound: -0.0378292, upper bound: 0.0369289
IS_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 7.35
Output dim: 3, lower bound: -0.0398111, upper bound: 0.0399960
IS_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 7.35
Output dim: 3, lower bound: -0.0399235, upper bound: 0.0396998
IS_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 7.35
Output dim: 3, lower bound: -0.0396987, upper bound: 0.0396987

## BFS IS instance: IS_B1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0338651, 0.0304485, -0.0075789, 0.0173805, -0.0512456, 0.0380274
1: -0.0250315, 0.0252274, -0.0083904, 0.0100560, -0.0350875, 0.0336178
2: -0.0601122, 0.0172336, -0.0156960, 0.0164559, -0.0765681, 0.0329296
3: 1.0027032, 1.0275360, 1.0053742, 1.0149827, -0.0122795, 0.0221618
4: -0.0112049, 0.0397189, -0.0042295, 0.0112521, -0.0224570, 0.0439484
5: -0.0129485, 0.0782295, -0.0013096, 0.0379702, -0.0509186, 0.0795391
6: -0.0588127, 0.0274977, -0.0250743, 0.0033066, -0.0621193, 0.0525720
7: -0.0483403, -0.0018225, -0.0286922, -0.0019058, -0.0463791, 0.0268272
8: -0.0268973, 0.0255479, -0.0134676, 0.0053612, -0.0322586, 0.0390155
9: -0.0266569, 0.0433844, -0.0140825, 0.0158113, -0.0424682, 0.0574669

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of IS_B1_B1_B1_A1_A1

### Relational analysis result of IS_B1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0385614, upper bound: 0.0381410
time: 2.31 seconds

## Relational analysis of IS_B1_B1_B1_A1_A2

### Relational analysis result of IS_B1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0383680, upper bound: 0.0380875
time: 2.34 seconds

## BFS IS instance: IS_B1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0379555, 0.0323062, -0.0070793, 0.0169544, -0.0549099, 0.0393855
1: -0.0280532, 0.0269627, -0.0080755, 0.0096517, -0.0377049, 0.0350382
2: -0.0655731, 0.0187125, -0.0144640, 0.0164316, -0.0820047, 0.0331765
3: 1.0020553, 1.0292304, 1.0054494, 1.0147861, -0.0127308, 0.0237809
4: -0.0125175, 0.0433838, -0.0042260, 0.0104838, -0.0230013, 0.0476098
5: -0.0153678, 0.0831881, -0.0012577, 0.0366586, -0.0520264, 0.0844459
6: -0.0629740, 0.0315754, -0.0243637, 0.0030659, -0.0660399, 0.0559391
7: -0.0509926, -0.0014486, -0.0281349, -0.0019096, -0.0490331, 0.0266764
8: -0.0294427, 0.0288304, -0.0132423, 0.0051858, -0.0346286, 0.0420727
9: -0.0284401, 0.0460723, -0.0137721, 0.0149090, -0.0433492, 0.0598444

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of IS_B1_B1_B1_A2_A1

### Relational analysis result of IS_B1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0385585, upper bound: 0.0381408
time: 2.39 seconds

## Relational analysis of IS_B1_B1_B1_A2_A2

### Relational analysis result of IS_B1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0383630, upper bound: 0.0380873
time: 2.30 seconds

## BFS IS instance: IS_B1_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0433746, 0.0346853, -0.0261096, 0.0306146, -0.0739892, 0.0607949
1: -0.0317748, 0.0288107, -0.0179355, 0.0259913, -0.0577661, 0.0467462
2: -0.0717756, 0.0214090, -0.0558374, 0.0171534, -0.0889290, 0.0772464
3: 1.0011249, 1.0308304, 1.0029805, 1.0276670, -0.0265422, 0.0278499
4: -0.0141613, 0.0482730, -0.0043287, 0.0391176, -0.0532789, 0.0526017
5: -0.0184035, 0.0892157, -0.0109862, 0.0773049, -0.0957084, 0.1002020
6: -0.0676719, 0.0369965, -0.0580286, 0.0115559, -0.0792278, 0.0950251
7: -0.0547764, -0.0016964, -0.0472593, -0.0018798, -0.0528408, 0.0455545
8: -0.0323285, 0.0330742, -0.0263418, 0.0141934, -0.0465218, 0.0594159
9: -0.0308064, 0.0492509, -0.0253462, 0.0431740, -0.0739803, 0.0745971

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=25, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=7, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_B1_B1_B2_B2_B1

### Relational analysis result of IS_B1_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0389173, upper bound: 0.0381754
time: 2.62 seconds

## Relational analysis of IS_B1_B1_B2_B2_B2

### Relational analysis result of IS_B1_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0389254, upper bound: 0.0381924
time: 2.22 seconds

## BFS IS instance: IS_B1_B2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0516845, 0.0413355, -0.0109940, 0.0198974, -0.0715819, 0.0523295
1: -0.0364409, 0.0316020, -0.0102430, 0.0127564, -0.0491972, 0.0418450
2: -0.0800625, 0.0276394, -0.0231229, 0.0166022, -0.0966647, 0.0507623
3: 0.9991066, 1.0328910, 1.0049424, 1.0167452, -0.0176386, 0.0279486
4: -0.0177127, 0.0570051, -0.0042506, 0.0162394, -0.0339520, 0.0612557
5: -0.0244428, 0.0968848, -0.0019434, 0.0456744, -0.0701172, 0.0988282
6: -0.0748349, 0.0466059, -0.0304295, 0.0047489, -0.0795839, 0.0770354
7: -0.0611100, -0.0014148, -0.0321205, -0.0023162, -0.0587456, 0.0307058
8: -0.0359786, 0.0423952, -0.0153606, 0.0065203, -0.0424989, 0.0577558
9: -0.0362328, 0.0539828, -0.0160696, 0.0211202, -0.0573530, 0.0700524

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=25, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 207

## Relational analysis of IS_B1_B2_B1_B1_A1

### Relational analysis result of IS_B1_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392640, upper bound: 0.0389339
time: 2.48 seconds

## Relational analysis of IS_B1_B2_B1_B1_A2

### Relational analysis result of IS_B1_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392640, upper bound: 0.0389339
time: 3.42 seconds

## BFS IS instance: IS_B1_B2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0494031, 0.0391030, -0.0127249, 0.0211417, -0.0705447, 0.0518280
1: -0.0351580, 0.0308061, -0.0111069, 0.0143605, -0.0495185, 0.0419130
2: -0.0777680, 0.0257450, -0.0269850, 0.0166598, -0.0944278, 0.0527300
3: 0.9997612, 1.0323695, 1.0046966, 1.0181364, -0.0183752, 0.0276729
4: -0.0166013, 0.0545153, -0.0042589, 0.0188851, -0.0354865, 0.0587742
5: -0.0227150, 0.0947759, -0.0029642, 0.0492277, -0.0719427, 0.0977401
6: -0.0728148, 0.0438278, -0.0336692, 0.0056171, -0.0784319, 0.0774970
7: -0.0593456, -0.0014937, -0.0338720, -0.0019661, -0.0573347, 0.0323783
8: -0.0350091, 0.0395063, -0.0167138, 0.0073999, -0.0424091, 0.0562201
9: -0.0346278, 0.0525857, -0.0171620, 0.0236212, -0.0582490, 0.0697476

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=25, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 207

## Relational analysis of IS_B1_B2_B1_B2_A1

### Relational analysis result of IS_B1_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392633, upper bound: 0.0389249
time: 2.20 seconds

## Relational analysis of IS_B1_B2_B1_B2_A2

### Relational analysis result of IS_B1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392633, upper bound: 0.0389249
time: 2.36 seconds

## BFS IS instance: IS_B1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0467224, 0.0365068, -0.0270260, 0.0313149, -0.0780373, 0.0635329
1: -0.0336488, 0.0300498, -0.0183402, 0.0271211, -0.0607699, 0.0483900
2: -0.0753659, 0.0233064, -0.0582184, 0.0171535, -0.0925194, 0.0815248
3: 1.0004604, 1.0319560, 1.0029664, 1.0288547, -0.0283943, 0.0289896
4: -0.0151691, 0.0517392, -0.0043254, 0.0405643, -0.0557334, 0.0560646
5: -0.0207374, 0.0923655, -0.0119797, 0.0789007, -0.0996381, 0.1043452
6: -0.0705550, 0.0406960, -0.0599806, 0.0122925, -0.0828476, 0.1006766
7: -0.0573178, -0.0016276, -0.0482256, -0.0022804, -0.0549841, 0.0465980
8: -0.0339913, 0.0361245, -0.0273320, 0.0148002, -0.0487914, 0.0634565
9: -0.0326715, 0.0511108, -0.0260031, 0.0443618, -0.0770333, 0.0771139

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=25, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=7, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_B1_B2_B2_B1_B1

### Relational analysis result of IS_B1_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0389617, upper bound: 0.0382480
time: 2.80 seconds

## Relational analysis of IS_B1_B2_B2_B1_B2

### Relational analysis result of IS_B1_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0389607, upper bound: 0.0382221
time: 3.27 seconds

## BFS IS instance: IS_B1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0472936, 0.0370852, -0.0278154, 0.0318601, -0.0791537, 0.0649006
1: -0.0339615, 0.0301887, -0.0187532, 0.0277531, -0.0617146, 0.0489419
2: -0.0758151, 0.0239063, -0.0598603, 0.0171909, -0.0930060, 0.0837665
3: 1.0003097, 1.0320348, 1.0028280, 1.0293405, -0.0290308, 0.0292068
4: -0.0155153, 0.0523088, -0.0043310, 0.0417421, -0.0572574, 0.0566398
5: -0.0212041, 0.0928043, -0.0123937, 0.0806005, -0.1018046, 0.1051980
6: -0.0710118, 0.0413852, -0.0613612, 0.0125960, -0.0836079, 0.1027464
7: -0.0577416, -0.0016185, -0.0490082, -0.0020203, -0.0556734, 0.0473897
8: -0.0342181, 0.0368840, -0.0278691, 0.0152070, -0.0494251, 0.0647531
9: -0.0331226, 0.0513666, -0.0264695, 0.0455488, -0.0786713, 0.0778360

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=25, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=8, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_B1_B2_B2_B2_A1

### Relational analysis result of IS_B1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0389862, upper bound: 0.0382408
time: 2.23 seconds

## Relational analysis of IS_B1_B2_B2_B2_A2

### Relational analysis result of IS_B1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0389742, upper bound: 0.0382407
time: 2.66 seconds

## BFS IS instance: IS_B2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0356266, 0.0311019, -0.0303903, 0.0293569, -0.0649835, 0.0614922
1: -0.0263864, 0.0258190, -0.0224821, 0.0241280, -0.0505144, 0.0483012
2: -0.0622174, 0.0177553, -0.0557899, 0.0171452, -0.0793626, 0.0735452
3: 1.0023699, 1.0280904, 1.0030990, 1.0267181, -0.0243483, 0.0249914
4: -0.0118320, 0.0411398, -0.0097361, 0.0370410, -0.0488730, 0.0508758
5: -0.0137676, 0.0802451, -0.0119325, 0.0741645, -0.0879321, 0.0921777
6: -0.0603432, 0.0292877, -0.0558394, 0.0235406, -0.0838837, 0.0851271
7: -0.0493124, -0.0013585, -0.0465110, -0.0020776, -0.0471972, 0.0450969
8: -0.0279086, 0.0269693, -0.0256777, 0.0223102, -0.0502188, 0.0526470
9: -0.0273643, 0.0443993, -0.0253832, 0.0409096, -0.0682739, 0.0697826

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 207

## Relational analysis of IS_B2_A1_B1_B1_A1

### Relational analysis result of IS_B2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392886, upper bound: 0.0392294
time: 2.33 seconds

## Relational analysis of IS_B2_A1_B1_B1_A2

### Relational analysis result of IS_B2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392886, upper bound: 0.0392294
time: 2.39 seconds

## BFS IS instance: IS_B2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0350110, 0.0308011, -0.0636612, 0.0422664, -0.0772774, 0.0944624
1: -0.0259276, 0.0255717, -0.0452113, 0.0399303, -0.0658579, 0.0707830
2: -0.0614167, 0.0174644, -0.1052149, 0.0178265, -0.0792432, 0.1226793
3: 1.0024751, 1.0278598, 1.0008194, 1.0419078, -0.0394326, 0.0270404
4: -0.0116306, 0.0405948, -0.0234414, 0.0659105, -0.0775411, 0.0640362
5: -0.0134016, 0.0794960, -0.0290478, 0.1125263, -0.1259279, 0.1085438
6: -0.0597341, 0.0286796, -0.0918929, 0.0647481, -0.1244822, 0.1205725
7: -0.0488895, -0.0014577, -0.0671865, -0.0021156, -0.0467486, 0.0656620
8: -0.0275301, 0.0264766, -0.0410606, 0.0557577, -0.0832878, 0.0675373
9: -0.0271017, 0.0439958, -0.0395025, 0.0676182, -0.0947199, 0.0834983

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B2_A1_B1_B2_B1

### Relational analysis result of IS_B2_A1_B1_B2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0367997, upper bound: 0.0360737
time: 2.07 seconds

## Relational analysis of IS_B2_A1_B1_B2_B2

### Relational analysis result of IS_B2_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392418, upper bound: 0.0391860
time: 2.42 seconds

## BFS IS instance: IS_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0070298, 0.0121351, -0.0222641, 0.0261860, -0.0324560, 0.0343992
1: -0.0051514, 0.0046063, -0.0169379, 0.0202566, -0.0254081, 0.0215442
2: 0.0008024, 0.0161590, -0.0436959, 0.0169848, -0.0161825, 0.0598549
3: 1.0063207, 1.0117761, 1.0036141, 1.0230033, -0.0166826, 0.0081620
4: -0.0041918, 0.0003044, -0.0064086, 0.0299548, -0.0341466, 0.0067130
5: -0.0014431, 0.0191093, -0.0077724, 0.0647303, -0.0656179, 0.0268817
6: -0.0146674, -0.0000675, -0.0470023, 0.0135308, -0.0281982, 0.0469348
7: -0.0208458, -0.0008925, -0.0414450, -0.0013731, -0.0194358, 0.0404674
8: -0.0125378, 0.0043813, -0.0219741, 0.0142979, -0.0268357, 0.0263554
9: -0.0095745, 0.0029382, -0.0219378, 0.0343592, -0.0439336, 0.0248760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=7, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 207

## Relational analysis of IS_B2_A1_B2_A1_A1

### Relational analysis result of IS_B2_A1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0363704, upper bound: 0.0369736
time: 1.90 seconds

## Relational analysis of IS_B2_A1_B2_A1_A2

### Relational analysis result of IS_B2_A1_B2_A1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0355731, upper bound: 0.0367105
time: 2.14 seconds

## BFS IS instance: IS_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0329298, 0.0300954, -0.0373917, 0.0320427, -0.0649725, 0.0674871
1: -0.0243679, 0.0247856, -0.0276178, 0.0268248, -0.0511927, 0.0524034
2: -0.0587500, 0.0172311, -0.0649699, 0.0184387, -0.0771886, 0.0822010
3: 1.0027629, 1.0270981, 1.0021429, 1.0291682, -0.0264053, 0.0249552
4: -0.0109435, 0.0388160, -0.0123334, 0.0429221, -0.0538656, 0.0511494
5: -0.0124749, 0.0769866, -0.0151934, 0.0824237, -0.0948985, 0.0921800
6: -0.0577649, 0.0266175, -0.0625385, 0.0310977, -0.0888625, 0.0891560
7: -0.0477997, -0.0012802, -0.0506347, -0.0013184, -0.0464660, 0.0492753
8: -0.0263875, 0.0248208, -0.0292295, 0.0284135, -0.0548010, 0.0540503
9: -0.0262270, 0.0426900, -0.0282458, 0.0456896, -0.0719166, 0.0709358

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_B2_A1_B2_A2_A1

### Relational analysis result of IS_B2_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0395899, upper bound: 0.0397213
time: 2.49 seconds

## Relational analysis of IS_B2_A1_B2_A2_A2

### Relational analysis result of IS_B2_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0395841, upper bound: 0.0397231
time: 2.45 seconds

## BFS IS instance: IS_B2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0228549, 0.0264362, -0.0068598, 0.0155764, -0.0384312, 0.0332959
1: -0.0173341, 0.0205543, -0.0068502, 0.0081186, -0.0254527, 0.0274045
2: -0.0446064, 0.0169962, -0.0097942, 0.0163910, -0.0609974, 0.0267904
3: 1.0035942, 1.0232956, 1.0056599, 1.0137392, -0.0101451, 0.0176357
4: -0.0066288, 0.0305060, -0.0042241, 0.0074522, -0.0140811, 0.0347301
5: -0.0080814, 0.0654782, -0.0013757, 0.0315528, -0.0396342, 0.0668538
6: -0.0476827, 0.0142075, -0.0215846, 0.0020075, -0.0496902, 0.0357921
7: -0.0418316, -0.0016882, -0.0259574, -0.0012936, -0.0404877, 0.0241980
8: -0.0222420, 0.0148647, -0.0130385, 0.0050896, -0.0273315, 0.0279032
9: -0.0221980, 0.0348639, -0.0125597, 0.0113456, -0.0335435, 0.0474236

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 156

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of IS_B2_A2_A1_B1_A1

### Relational analysis result of IS_B2_A2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0371596, upper bound: 0.0361631
time: 2.11 seconds

## Relational analysis of IS_B2_A2_A1_B1_A2

### Relational analysis result of IS_B2_A2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0370616, upper bound: 0.0361624
time: 2.52 seconds

## BFS IS instance: IS_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0379009, 0.0323111, -0.0516712, 0.0413920, -0.0792928, 0.0839823
1: -0.0279783, 0.0270702, -0.0364329, 0.0316699, -0.0596482, 0.0635031
2: -0.0657127, 0.0186259, -0.0801636, 0.0275899, -0.0933026, 0.0987894
3: 1.0021034, 1.0294116, 0.9991136, 1.0329449, -0.0308415, 0.0302979
4: -0.0124721, 0.0434293, -0.0176852, 0.0570616, -0.0695337, 0.0611144
5: -0.0155162, 0.0831194, -0.0244838, 0.0968781, -0.1123943, 0.1076032
6: -0.0631241, 0.0315606, -0.0748877, 0.0466693, -0.1097933, 0.1064483
7: -0.0510108, -0.0016212, -0.0611289, -0.0013028, -0.0497080, 0.0594556
8: -0.0295309, 0.0288042, -0.0360273, 0.0424394, -0.0719703, 0.0648315
9: -0.0284791, 0.0460846, -0.0362279, 0.0540345, -0.0825136, 0.0823125

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 13

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_B2_A2_A1_B2_B1

### Relational analysis result of IS_B2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0395870, upper bound: 0.0398363
time: 3.08 seconds

## Relational analysis of IS_B2_A2_A1_B2_B2

### Relational analysis result of IS_B2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0395869, upper bound: 0.0397842
time: 2.40 seconds

## BFS IS instance: IS_B2_A2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0415915, 0.0339946, -0.0569805, 0.0466818, -0.0882733, 0.0909751
1: -0.0307713, 0.0282034, -0.0394046, 0.0335843, -0.0643556, 0.0676081
2: -0.0699253, 0.0205371, -0.0855679, 0.0320330, -0.1019584, 0.1061050
3: 1.0014579, 1.0302820, 0.9975755, 1.0342342, -0.0327762, 0.0327065
4: -0.0137371, 0.0464886, -0.0202881, 0.0629062, -0.0766433, 0.0667767
5: -0.0173066, 0.0874923, -0.0286169, 0.1017581, -0.1190647, 0.1161092
6: -0.0662403, 0.0351446, -0.0796588, 0.0531967, -0.1194369, 0.1148034
7: -0.0534448, -0.0015942, -0.0652419, -0.0011105, -0.0523343, 0.0636192
8: -0.0314902, 0.0316921, -0.0383499, 0.0492394, -0.0807296, 0.0700421
9: -0.0299258, 0.0483074, -0.0400055, 0.0573071, -0.0872329, 0.0883129

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 1

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_B2_A2_A2_A1_A1

### Relational analysis result of IS_B2_A2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393301, upper bound: 0.0393989
time: 2.66 seconds

## Relational analysis of IS_B2_A2_A2_A1_A2

### Relational analysis result of IS_B2_A2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0397541, upper bound: 0.0395347
time: 2.33 seconds

## BFS IS instance: IS_B2_A2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0947070, 0.0535030, -0.0561572, 0.0458352, -0.1405422, 0.1096603
1: -0.0605391, 0.0445843, -0.0389444, 0.0332543, -0.0937934, 0.0835287
2: -0.1217332, 0.0476899, -0.0846751, 0.0313676, -0.1531008, 0.1323649
3: 0.9886540, 1.0449296, 0.9978117, 1.0340009, -0.0453469, 0.0471179
4: -0.0267538, 0.0985153, -0.0198965, 0.0619702, -0.0887239, 0.1184118
5: -0.0492836, 0.1362204, -0.0279548, 0.1009864, -0.1502700, 0.1641752
6: -0.1057482, 0.0910844, -0.0788864, 0.0521609, -0.1579091, 0.1699708
7: -0.0926568, 0.0070837, -0.0645941, -0.0012643, -0.0913925, 0.0716778
8: -0.0558075, 0.0733113, -0.0379677, 0.0481673, -0.1039748, 0.1112791
9: -0.0558071, 0.0744873, -0.0394197, 0.0567664, -0.1125735, 0.1139070

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_B2_A2_A2_A2_A1

### Relational analysis result of IS_B2_A2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0383668, upper bound: 0.0391210
time: 2.05 seconds

## Relational analysis of IS_B2_A2_A2_A2_A2

### Relational analysis result of IS_B2_A2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0383668, upper bound: 0.0383668
time: 2.43 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 5.78 seconds
IS_B1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 5.78
Output dim: 3, lower bound: -0.0385614, upper bound: 0.0381410
IS_B1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 5.78
Output dim: 3, lower bound: -0.0383680, upper bound: 0.0380875
IS_B1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 5.78
Output dim: 3, lower bound: -0.0385585, upper bound: 0.0381408
IS_B1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 5.78
Output dim: 3, lower bound: -0.0383630, upper bound: 0.0380873
IS_B1_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 5.78
Output dim: 3, lower bound: -0.0389173, upper bound: 0.0381754
IS_B1_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 5.78
Output dim: 3, lower bound: -0.0389254, upper bound: 0.0381924
IS_B1_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.78
Output dim: 3, lower bound: -0.0392640, upper bound: 0.0389339
IS_B1_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.78
Output dim: 3, lower bound: -0.0392640, upper bound: 0.0389339
IS_B1_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.78
Output dim: 3, lower bound: -0.0392633, upper bound: 0.0389249
IS_B1_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.78
Output dim: 3, lower bound: -0.0392633, upper bound: 0.0389249
IS_B1_B2_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 5.78
Output dim: 3, lower bound: -0.0389617, upper bound: 0.0382480
IS_B1_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 5.78
Output dim: 3, lower bound: -0.0389607, upper bound: 0.0382221
IS_B1_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.78
Output dim: 3, lower bound: -0.0389862, upper bound: 0.0382408
IS_B1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.78
Output dim: 3, lower bound: -0.0389742, upper bound: 0.0382407
IS_B2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.78
Output dim: 3, lower bound: -0.0392886, upper bound: 0.0392294
IS_B2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.78
Output dim: 3, lower bound: -0.0392886, upper bound: 0.0392294
IS_B2_A1_B1_B2_B1, status: Status.VERIFIED, split count: 5, time: 5.78
Output dim: 3, lower bound: -0.0367997, upper bound: 0.0360737
IS_B2_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 5.78
Output dim: 3, lower bound: -0.0392418, upper bound: 0.0391860
IS_B2_A1_B2_A1_A1, status: Status.VERIFIED, split count: 5, time: 5.78
Output dim: 3, lower bound: -0.0363704, upper bound: 0.0369736
IS_B2_A1_B2_A1_A2, status: Status.VERIFIED, split count: 5, time: 5.78
Output dim: 3, lower bound: -0.0355731, upper bound: 0.0367105
IS_B2_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 5.78
Output dim: 3, lower bound: -0.0395899, upper bound: 0.0397213
IS_B2_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 5.78
Output dim: 3, lower bound: -0.0395841, upper bound: 0.0397231
IS_B2_A2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 5.78
Output dim: 3, lower bound: -0.0371596, upper bound: 0.0361631
IS_B2_A2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 5.78
Output dim: 3, lower bound: -0.0370616, upper bound: 0.0361624
IS_B2_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 5.78
Output dim: 3, lower bound: -0.0395870, upper bound: 0.0398363
IS_B2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 5.78
Output dim: 3, lower bound: -0.0395869, upper bound: 0.0397842
IS_B2_A2_A2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 5.78
Output dim: 3, lower bound: -0.0393301, upper bound: 0.0393989
IS_B2_A2_A2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 5.78
Output dim: 3, lower bound: -0.0397541, upper bound: 0.0395347
IS_B2_A2_A2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 5.78
Output dim: 3, lower bound: -0.0383668, upper bound: 0.0391210
IS_B2_A2_A2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 5.78
Output dim: 3, lower bound: -0.0383668, upper bound: 0.0383668

## BFS IS instance: IS_B1_B1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0180227, 0.0241764, -0.0068967, 0.0167275, -0.0347503, 0.0310731
1: -0.0142838, 0.0172305, -0.0079352, 0.0093934, -0.0236772, 0.0251656
2: -0.0360017, 0.0169461, -0.0137659, 0.0164258, -0.0524276, 0.0307119
3: 1.0038850, 1.0195788, 1.0054823, 1.0146065, -0.0107214, 0.0140965
4: -0.0050164, 0.0255386, -0.0042255, 0.0101274, -0.0151438, 0.0297642
5: -0.0042598, 0.0597600, -0.0012177, 0.0360801, -0.0403399, 0.0609777
6: -0.0410256, 0.0082647, -0.0240474, 0.0028617, -0.0438873, 0.0323122
7: -0.0384253, -0.0018648, -0.0278671, -0.0019092, -0.0364789, 0.0259577
8: -0.0189425, 0.0101992, -0.0130943, 0.0050885, -0.0240310, 0.0232935
9: -0.0197056, 0.0307044, -0.0136099, 0.0144892, -0.0341947, 0.0443142

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 229

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of IS_B1_B1_B1_A1_A1_B1

### Relational analysis result of IS_B1_B1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0383680, upper bound: 0.0380875
time: 3.36 seconds

## Relational analysis of IS_B1_B1_B1_A1_A1_B2

### Relational analysis result of IS_B1_B1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0383680, upper bound: 0.0380875
time: 2.13 seconds

## BFS IS instance: IS_B1_B1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0121776, 0.0205639, -0.0064597, 0.0138134, -0.0259910, 0.0268521
1: -0.0110524, 0.0127203, -0.0054086, 0.0063918, -0.0174442, 0.0181289
2: -0.0244187, 0.0167888, -0.0041699, 0.0162354, -0.0406541, 0.0209587
3: 1.0044386, 1.0158365, 1.0061179, 1.0127951, -0.0083565, 0.0097185
4: -0.0042903, 0.0178412, -0.0042000, 0.0034130, -0.0077034, 0.0220413
5: -0.0017478, 0.0491957, -0.0010405, 0.0245996, -0.0263474, 0.0502362
6: -0.0317004, 0.0042026, -0.0178121, 0.0010739, -0.0327744, 0.0220147
7: -0.0331999, -0.0012992, -0.0231167, -0.0019355, -0.0312298, 0.0217839
8: -0.0152768, 0.0068784, -0.0126776, 0.0038719, -0.0191487, 0.0195561
9: -0.0164820, 0.0233798, -0.0110309, 0.0065122, -0.0229942, 0.0344107

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 207

## Relational analysis of IS_B1_B1_B1_A1_A2_A1

### Relational analysis result of IS_B1_B1_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0383680, upper bound: 0.0380875
time: 2.34 seconds

## Relational analysis of IS_B1_B1_B1_A1_A2_A2

### Relational analysis result of IS_B1_B1_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0383680, upper bound: 0.0380875
time: 2.16 seconds

## BFS IS instance: IS_B1_B1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0214768, 0.0255405, -0.0066705, 0.0163946, -0.0378714, 0.0322110
1: -0.0166079, 0.0189620, -0.0076196, 0.0090733, -0.0256811, 0.0265815
2: -0.0412403, 0.0170174, -0.0127064, 0.0164016, -0.0576418, 0.0297238
3: 1.0036145, 1.0212578, 1.0055574, 1.0144354, -0.0108209, 0.0157003
4: -0.0064272, 0.0285483, -0.0042221, 0.0093634, -0.0157906, 0.0327704
5: -0.0060155, 0.0637137, -0.0011824, 0.0347610, -0.0407765, 0.0648962
6: -0.0447880, 0.0125806, -0.0233328, 0.0026928, -0.0474808, 0.0359134
7: -0.0405940, -0.0015429, -0.0273273, -0.0019129, -0.0386465, 0.0257441
8: -0.0205802, 0.0136091, -0.0130383, 0.0049302, -0.0255104, 0.0266474
9: -0.0211712, 0.0334895, -0.0133211, 0.0135833, -0.0347544, 0.0468106

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 229

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of IS_B1_B1_B1_A2_A1_B1

### Relational analysis result of IS_B1_B1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0383630, upper bound: 0.0380873
time: 2.19 seconds

## Relational analysis of IS_B1_B1_B1_A2_A1_B2

### Relational analysis result of IS_B1_B1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0383630, upper bound: 0.0380873
time: 2.62 seconds

## BFS IS instance: IS_B1_B1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0141626, 0.0219818, -0.0064558, 0.0134927, -0.0276553, 0.0282076
1: -0.0120452, 0.0145346, -0.0052770, 0.0060863, -0.0181315, 0.0198116
2: -0.0288159, 0.0168581, -0.0031583, 0.0162105, -0.0450264, 0.0200164
3: 1.0041575, 1.0173934, 1.0061878, 1.0126445, -0.0084870, 0.0112056
4: -0.0043001, 0.0208566, -0.0041965, 0.0026740, -0.0069741, 0.0250531
5: -0.0025568, 0.0532740, -0.0010333, 0.0233215, -0.0258782, 0.0543072
6: -0.0353791, 0.0051815, -0.0171212, 0.0009156, -0.0362948, 0.0223027
7: -0.0351990, -0.0009733, -0.0226012, -0.0019392, -0.0332263, 0.0215971
8: -0.0168006, 0.0078057, -0.0126224, 0.0037702, -0.0205708, 0.0204282
9: -0.0177252, 0.0262476, -0.0107575, 0.0056380, -0.0233631, 0.0370051

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 156

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 207

## Relational analysis of IS_B1_B1_B1_A2_A2_A1

### Relational analysis result of IS_B1_B1_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0383630, upper bound: 0.0380873
time: 2.75 seconds

## Relational analysis of IS_B1_B1_B1_A2_A2_A2

### Relational analysis result of IS_B1_B1_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0383630, upper bound: 0.0380873
time: 2.41 seconds

## BFS IS instance: IS_B1_B1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0392514, 0.0329104, -0.0218776, 0.0276241, -0.0668755, 0.0547880
1: -0.0290535, 0.0273524, -0.0157712, 0.0223366, -0.0513901, 0.0431236
2: -0.0670769, 0.0193713, -0.0467453, 0.0169898, -0.0840667, 0.0661167
3: 1.0018688, 1.0295055, 1.0035412, 1.0246814, -0.0228126, 0.0259643
4: -0.0129675, 0.0444545, -0.0043045, 0.0327272, -0.0456946, 0.0487590
5: -0.0159584, 0.0847866, -0.0084543, 0.0684115, -0.0843699, 0.0932409
6: -0.0640934, 0.0328026, -0.0503454, 0.0097007, -0.0737942, 0.0831480
7: -0.0518558, -0.0018017, -0.0430303, -0.0019405, -0.0498605, 0.0412035
8: -0.0301078, 0.0298279, -0.0232956, 0.0119794, -0.0420873, 0.0531234
9: -0.0289396, 0.0468796, -0.0227635, 0.0369783, -0.0659178, 0.0696431

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=24, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_B1_B1_B2_B2_B1_A1

### Relational analysis result of IS_B1_B1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0383847, upper bound: 0.0381755
time: 2.91 seconds

## Relational analysis of IS_B1_B1_B2_B2_B1_A2

### Relational analysis result of IS_B1_B1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0383847, upper bound: 0.0381754
time: 2.38 seconds

## BFS IS instance: IS_B1_B1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0376883, 0.0321577, -0.0235690, 0.0288303, -0.0665186, 0.0557267
1: -0.0278754, 0.0267727, -0.0166220, 0.0238704, -0.0517459, 0.0433947
2: -0.0651119, 0.0186157, -0.0504677, 0.0170506, -0.0821624, 0.0690834
3: 1.0021175, 1.0289984, 1.0032989, 1.0260026, -0.0238851, 0.0256995
4: -0.0124495, 0.0430977, -0.0043129, 0.0352911, -0.0477406, 0.0474106
5: -0.0150953, 0.0828726, -0.0095685, 0.0718954, -0.0869907, 0.0924410
6: -0.0626101, 0.0312829, -0.0534687, 0.0105209, -0.0731310, 0.0847516
7: -0.0507921, -0.0018366, -0.0447233, -0.0015685, -0.0491734, 0.0428626
8: -0.0291978, 0.0285968, -0.0245937, 0.0129158, -0.0421136, 0.0531905
9: -0.0282946, 0.0458707, -0.0238194, 0.0394335, -0.0677282, 0.0696901

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=24, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_B1_B1_B2_B2_B2_A1

### Relational analysis result of IS_B1_B1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0383900, upper bound: 0.0381924
time: 3.90 seconds

## Relational analysis of IS_B1_B1_B2_B2_B2_A2

### Relational analysis result of IS_B1_B1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0383900, upper bound: 0.0381925
time: 2.21 seconds

## BFS IS instance: IS_B1_B2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0496914, 0.0392799, -0.0109940, 0.0198974, -0.0695888, 0.0502739
1: -0.0353257, 0.0308235, -0.0102430, 0.0127564, -0.0480820, 0.0410665
2: -0.0779357, 0.0259950, -0.0231229, 0.0166022, -0.0945379, 0.0491179
3: 0.9996828, 1.0323477, 1.0049424, 1.0167452, -0.0170624, 0.0274053
4: -0.0167427, 0.0547594, -0.0042506, 0.0162394, -0.0329820, 0.0590100
5: -0.0228478, 0.0950314, -0.0019434, 0.0456744, -0.0685222, 0.0969748
6: -0.0729763, 0.0441072, -0.0304295, 0.0047489, -0.0777253, 0.0745367
7: -0.0595467, -0.0017613, -0.0321205, -0.0023162, -0.0571823, 0.0303592
8: -0.0350646, 0.0397902, -0.0153606, 0.0065203, -0.0415849, 0.0551509
9: -0.0348021, 0.0526964, -0.0160696, 0.0211202, -0.0559223, 0.0687660

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=25, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_B1_B2_B1_B1_A1_A1

### Relational analysis result of IS_B1_B2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0389513, upper bound: 0.0386831
time: 3.01 seconds

## Relational analysis of IS_B1_B2_B1_B1_A1_A2

### Relational analysis result of IS_B1_B2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0389513, upper bound: 0.0389339
time: 2.47 seconds

## BFS IS instance: IS_B1_B2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.1065072, 0.0968241, -0.0109940, 0.0198974, -0.1264046, 0.1078181
1: -0.0671680, 0.0533452, -0.0102430, 0.0127564, -0.0799244, 0.0635882
2: -0.1392884, 0.0712754, -0.0231229, 0.0166022, -0.1558906, 0.0943983
3: 0.9834775, 1.0479988, 1.0049424, 1.0167452, -0.0332677, 0.0430564
4: -0.0433569, 0.1191890, -0.0042506, 0.0162394, -0.0595963, 0.1234396
5: -0.0677913, 0.1486176, -0.0019434, 0.0456744, -0.1134657, 0.1505609
6: -0.1258620, 0.1151651, -0.0304295, 0.0047489, -0.1306109, 0.1455946
7: -0.1042214, 0.0101256, -0.0321205, -0.0023162, -0.1018524, 0.0422462
8: -0.0610823, 0.1130538, -0.0153606, 0.0065203, -0.0676026, 0.1284144
9: -0.0748080, 0.0900353, -0.0160696, 0.0211202, -0.0959282, 0.1061049

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=25, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_B1_B2_B1_B1_A2_A1

### Relational analysis result of IS_B1_B2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0389513, upper bound: 0.0386269
time: 23.58 seconds

## Relational analysis of IS_B1_B2_B1_B1_A2_A2

### Relational analysis result of IS_B1_B2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0389513, upper bound: 0.0389339
time: 2.31 seconds

## BFS IS instance: IS_B1_B2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0474122, 0.0370471, -0.0127249, 0.0211417, -0.0685538, 0.0497720
1: -0.0340459, 0.0300275, -0.0111069, 0.0143605, -0.0484063, 0.0411344
2: -0.0756467, 0.0240941, -0.0269850, 0.0166598, -0.0923064, 0.0510790
3: 1.0002836, 1.0318249, 1.0046966, 1.0181364, -0.0178528, 0.0271283
4: -0.0156317, 0.0522701, -0.0042589, 0.0188851, -0.0345169, 0.0565290
5: -0.0211153, 0.0929324, -0.0029642, 0.0492277, -0.0703430, 0.0958965
6: -0.0709598, 0.0413287, -0.0336692, 0.0056171, -0.0765769, 0.0749979
7: -0.0577821, -0.0018261, -0.0338720, -0.0019661, -0.0557716, 0.0320459
8: -0.0340915, 0.0369022, -0.0167138, 0.0073999, -0.0414914, 0.0536160
9: -0.0331960, 0.0513063, -0.0171620, 0.0236212, -0.0568172, 0.0684683

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=25, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_B1_B2_B1_B2_A1_A1

### Relational analysis result of IS_B1_B2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0389567, upper bound: 0.0386862
time: 2.13 seconds

## Relational analysis of IS_B1_B2_B1_B2_A1_A2

### Relational analysis result of IS_B1_B2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0389567, upper bound: 0.0389249
time: 2.55 seconds

## BFS IS instance: IS_B1_B2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.1042994, 0.0945266, -0.0127249, 0.0211417, -0.1254410, 0.1072515
1: -0.0659326, 0.0524338, -0.0111069, 0.0143605, -0.0802931, 0.0635407
2: -0.1368476, 0.0695139, -0.0269850, 0.0166598, -0.1535074, 0.0964989
3: 0.9841133, 1.0473675, 1.0046966, 1.0181364, -0.0340231, 0.0426708
4: -0.0423211, 0.1166416, -0.0042589, 0.0188851, -0.0612062, 0.1209005
5: -0.0660049, 0.1465212, -0.0029642, 0.0492277, -0.1152326, 0.1494853
6: -0.1237592, 0.1123745, -0.0336692, 0.0056171, -0.1293763, 0.1460437
7: -0.1024688, 0.0095346, -0.0338720, -0.0019661, -0.1004512, 0.0434066
8: -0.0600514, 0.1101569, -0.0167138, 0.0073999, -0.0674514, 0.1268706
9: -0.0732321, 0.0885488, -0.0171620, 0.0236212, -0.0968533, 0.1057107

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=25, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_B1_B2_B1_B2_A2_A1

### Relational analysis result of IS_B1_B2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0389567, upper bound: 0.0386862
time: 4.08 seconds

## Relational analysis of IS_B1_B2_B1_B2_A2_A2

### Relational analysis result of IS_B1_B2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0389567, upper bound: 0.0389249
time: 2.23 seconds

## BFS IS instance: IS_B1_B2_B2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0419484, 0.0342181, -0.0230304, 0.0284927, -0.0704411, 0.0572485
1: -0.0309751, 0.0285149, -0.0162998, 0.0236628, -0.0546379, 0.0448147
2: -0.0706443, 0.0206016, -0.0496224, 0.0169951, -0.0876394, 0.0702239
3: 1.0014578, 1.0306427, 1.0034908, 1.0260203, -0.0245625, 0.0271519
4: -0.0137981, 0.0469483, -0.0043021, 0.0345304, -0.0483285, 0.0512505
5: -0.0177014, 0.0879790, -0.0095762, 0.0705162, -0.0882176, 0.0975551
6: -0.0668446, 0.0355162, -0.0527207, 0.0105316, -0.0773762, 0.0882369
7: -0.0537602, -0.0017447, -0.0442298, -0.0023444, -0.0513642, 0.0424753
8: -0.0318012, 0.0319903, -0.0244290, 0.0127012, -0.0445024, 0.0564193
9: -0.0301521, 0.0486431, -0.0235622, 0.0385131, -0.0686652, 0.0722054

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=24, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=7, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_B1_B2_B2_B1_B1_A1

### Relational analysis result of IS_B1_B2_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0386284, upper bound: 0.0380903
time: 3.03 seconds

## Relational analysis of IS_B1_B2_B2_B1_B1_A2

### Relational analysis result of IS_B1_B2_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0386284, upper bound: 0.0382473
time: 3.04 seconds

## BFS IS instance: IS_B1_B2_B2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0401811, 0.0334109, -0.0245616, 0.0295819, -0.0697630, 0.0579726
1: -0.0296952, 0.0279047, -0.0170731, 0.0250262, -0.0547215, 0.0449778
2: -0.0685537, 0.0197200, -0.0529589, 0.0170551, -0.0856088, 0.0726789
3: 1.0017543, 1.0301172, 1.0032692, 1.0271891, -0.0254349, 0.0268480
4: -0.0132290, 0.0454054, -0.0043104, 0.0368464, -0.0500754, 0.0497158
5: -0.0167407, 0.0859282, -0.0105591, 0.0736949, -0.0904356, 0.0964873
6: -0.0652794, 0.0337676, -0.0555264, 0.0112504, -0.0765298, 0.0892940
7: -0.0525525, -0.0017853, -0.0457582, -0.0020149, -0.0504894, 0.0439607
8: -0.0308252, 0.0305984, -0.0255862, 0.0135533, -0.0443785, 0.0561846
9: -0.0294123, 0.0475697, -0.0245111, 0.0407524, -0.0701648, 0.0720808

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=24, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_B1_B2_B2_B1_B2_A1

### Relational analysis result of IS_B1_B2_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0386354, upper bound: 0.0380909
time: 2.37 seconds

## Relational analysis of IS_B1_B2_B2_B1_B2_A2

### Relational analysis result of IS_B1_B2_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0386354, upper bound: 0.0382221
time: 2.60 seconds

## BFS IS instance: IS_B1_B2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0363325, 0.0314910, -0.0263180, 0.0308031, -0.0671356, 0.0578090
1: -0.0268623, 0.0262370, -0.0179872, 0.0264646, -0.0533269, 0.0442241
2: -0.0633627, 0.0179728, -0.0566467, 0.0171320, -0.0804946, 0.0746195
3: 1.0023487, 1.0285304, 1.0030247, 1.0282885, -0.0259398, 0.0255057
4: -0.0120075, 0.0418936, -0.0043224, 0.0394842, -0.0514917, 0.0462160
5: -0.0143200, 0.0812028, -0.0115015, 0.0774505, -0.0917705, 0.0927043
6: -0.0612670, 0.0299597, -0.0586496, 0.0119422, -0.0732092, 0.0886093
7: -0.0498577, -0.0019027, -0.0475124, -0.0020434, -0.0477681, 0.0455841
8: -0.0283878, 0.0275238, -0.0267864, 0.0144231, -0.0428110, 0.0543103
9: -0.0277178, 0.0449821, -0.0255562, 0.0433528, -0.0710706, 0.0705382

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=25, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_B1_B2_B2_B2_A1_A1

### Relational analysis result of IS_B1_B2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0386315, upper bound: 0.0381076
time: 2.31 seconds

## Relational analysis of IS_B1_B2_B2_B2_A1_A2

### Relational analysis result of IS_B1_B2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0386315, upper bound: 0.0382408
time: 2.75 seconds

## BFS IS instance: IS_B1_B2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0401483, 0.0333974, -0.0255657, 0.0302717, -0.0704200, 0.0589631
1: -0.0296693, 0.0279225, -0.0176037, 0.0258050, -0.0554744, 0.0455262
2: -0.0685386, 0.0197218, -0.0550180, 0.0171045, -0.0856431, 0.0747399
3: 1.0017339, 1.0301658, 1.0031197, 1.0277522, -0.0260183, 0.0270461
4: -0.0132172, 0.0453778, -0.0043183, 0.0383467, -0.0515638, 0.0496961
5: -0.0167832, 0.0858483, -0.0110446, 0.0758838, -0.0926670, 0.0968929
6: -0.0652589, 0.0337637, -0.0572755, 0.0116076, -0.0768665, 0.0910392
7: -0.0525241, -0.0015092, -0.0467557, -0.0020465, -0.0504358, 0.0452342
8: -0.0308687, 0.0305849, -0.0262385, 0.0140338, -0.0449024, 0.0568234
9: -0.0294071, 0.0475247, -0.0250955, 0.0422628, -0.0716699, 0.0726202

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=25, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_B1_B2_B2_B2_A2_A1

### Relational analysis result of IS_B1_B2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0386382, upper bound: 0.0381128
time: 2.23 seconds

## Relational analysis of IS_B1_B2_B2_B2_A2_A2

### Relational analysis result of IS_B1_B2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0386382, upper bound: 0.0382406
time: 2.40 seconds

## BFS IS instance: IS_B2_A1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0341232, 0.0304605, -0.0303903, 0.0293569, -0.0634800, 0.0608508
1: -0.0252611, 0.0252430, -0.0224821, 0.0241280, -0.0493891, 0.0477251
2: -0.0602905, 0.0172487, -0.0557899, 0.0171452, -0.0774357, 0.0730386
3: 1.0026259, 1.0275501, 1.0030990, 1.0267181, -0.0240922, 0.0244511
4: -0.0113345, 0.0398361, -0.0097361, 0.0370410, -0.0483755, 0.0495722
5: -0.0129757, 0.0784244, -0.0119325, 0.0741645, -0.0871402, 0.0903569
6: -0.0589006, 0.0278033, -0.0558394, 0.0235406, -0.0824412, 0.0836427
7: -0.0483775, -0.0016588, -0.0465110, -0.0020776, -0.0462619, 0.0447764
8: -0.0270290, 0.0257685, -0.0256777, 0.0223102, -0.0493392, 0.0514462
9: -0.0267323, 0.0434313, -0.0253832, 0.0409096, -0.0676419, 0.0688145

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_B2_A1_B1_B1_A1_A1

### Relational analysis result of IS_B2_A1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0380114, upper bound: 0.0387253
time: 2.37 seconds

## Relational analysis of IS_B2_A1_B1_B1_A1_A2

### Relational analysis result of IS_B2_A1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0380114, upper bound: 0.0394151
time: 2.26 seconds

## BFS IS instance: IS_B2_A1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0758160, 0.0507242, -0.0303903, 0.0293569, -0.1051729, 0.0811145
1: -0.0563748, 0.0418154, -0.0224821, 0.0241280, -0.0805028, 0.0642976
2: -0.1142714, 0.0367190, -0.0557899, 0.0171452, -0.1314166, 0.0925089
3: 0.9953100, 1.0427121, 1.0030990, 1.0267181, -0.0314081, 0.0396131
4: -0.0249495, 0.0767155, -0.0097361, 0.0370410, -0.0619906, 0.0864516
5: -0.0373767, 0.1293332, -0.0119325, 0.0741645, -0.1115412, 0.1412657
6: -0.1000231, 0.0687778, -0.0558394, 0.0235406, -0.1235636, 0.1246172
7: -0.0769322, 0.0051498, -0.0465110, -0.0020776, -0.0748103, 0.0516608
8: -0.0522962, 0.0590409, -0.0256777, 0.0223102, -0.0746065, 0.0847186
9: -0.0444570, 0.0707396, -0.0253832, 0.0409096, -0.0853666, 0.0961228

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=26, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=7, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_B2_A1_B1_B1_A2_A1

### Relational analysis result of IS_B2_A1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0380114, upper bound: 0.0387253
time: 2.36 seconds

## Relational analysis of IS_B2_A1_B1_B1_A2_A2

### Relational analysis result of IS_B2_A1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0380114, upper bound: 0.0394151
time: 2.27 seconds

## BFS IS instance: IS_B2_A1_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0350110, 0.0308011, -0.0604084, 0.0410082, -0.0760192, 0.0912095
1: -0.0259276, 0.0255717, -0.0429872, 0.0383898, -0.0643174, 0.0685588
2: -0.0614167, 0.0174644, -0.1003889, 0.0177591, -0.0791758, 0.1178532
3: 1.0024751, 1.0278598, 1.0010394, 1.0404238, -0.0379486, 0.0268204
4: -0.0116306, 0.0405948, -0.0220959, 0.0630982, -0.0747288, 0.0626907
5: -0.0134016, 0.0794960, -0.0273781, 0.1087864, -0.1221880, 0.1068740
6: -0.0597341, 0.0286796, -0.0883829, 0.0607078, -0.1204419, 0.1170624
7: -0.0488895, -0.0014577, -0.0651684, -0.0021200, -0.0467409, 0.0636347
8: -0.0275301, 0.0264766, -0.0395576, 0.0524720, -0.0800021, 0.0660342
9: -0.0271017, 0.0439958, -0.0381252, 0.0650115, -0.0921132, 0.0821209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=25, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 108

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_B2_A1_B1_B2_B2_B1

### Relational analysis result of IS_B2_A1_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0391260, upper bound: 0.0390701
time: 2.35 seconds

## Relational analysis of IS_B2_A1_B1_B2_B2_B2

### Relational analysis result of IS_B2_A1_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0391260, upper bound: 0.0390486
time: 2.99 seconds

## BFS IS instance: IS_B2_A1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0249382, 0.0270597, -0.0339121, 0.0305188, -0.0554570, 0.0609718
1: -0.0188652, 0.0210668, -0.0250209, 0.0254511, -0.0443164, 0.0460877
2: -0.0470254, 0.0170637, -0.0604583, 0.0172290, -0.0642545, 0.0775220
3: 1.0033867, 1.0235145, 1.0026847, 1.0278839, -0.0244972, 0.0208298
4: -0.0076761, 0.0319247, -0.0111928, 0.0398708, -0.0475470, 0.0431175
5: -0.0084497, 0.0677868, -0.0133020, 0.0781961, -0.0866457, 0.0810888
6: -0.0492252, 0.0168320, -0.0591317, 0.0276628, -0.0768880, 0.0759637
7: -0.0429296, -0.0013446, -0.0484341, -0.0013752, -0.0415056, 0.0470090
8: -0.0226777, 0.0169670, -0.0271725, 0.0256408, -0.0483185, 0.0441395
9: -0.0228564, 0.0363998, -0.0267667, 0.0434398, -0.0662961, 0.0631665

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 207

## Relational analysis of IS_B2_A1_B2_A2_A1_A1

### Relational analysis result of IS_B2_A1_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0394573, upper bound: 0.0393771
time: 2.53 seconds

## Relational analysis of IS_B2_A1_B2_A2_A1_A2

### Relational analysis result of IS_B2_A1_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0391354, upper bound: 0.0393532
time: 2.19 seconds

## BFS IS instance: IS_B2_A1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0285217, 0.0284626, -0.0323816, 0.0300396, -0.0585613, 0.0608441
1: -0.0213027, 0.0228293, -0.0238880, 0.0248835, -0.0461863, 0.0467173
2: -0.0524241, 0.0171364, -0.0584771, 0.0172015, -0.0696256, 0.0756134
3: 1.0031083, 1.0252582, 1.0028800, 1.0273538, -0.0242455, 0.0223782
4: -0.0091461, 0.0350577, -0.0106512, 0.0385880, -0.0477341, 0.0457089
5: -0.0103855, 0.0718776, -0.0126994, 0.0763592, -0.0867447, 0.0845770
6: -0.0531728, 0.0213067, -0.0576944, 0.0260766, -0.0792494, 0.0790010
7: -0.0451658, -0.0009679, -0.0476593, -0.0013851, -0.0437349, 0.0466133
8: -0.0244007, 0.0205529, -0.0264249, 0.0243571, -0.0487578, 0.0469779
9: -0.0243992, 0.0392700, -0.0261364, 0.0424295, -0.0668287, 0.0654064

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 207

## Relational analysis of IS_B2_A1_B2_A2_A2_A1

### Relational analysis result of IS_B2_A1_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0394584, upper bound: 0.0393807
time: 2.67 seconds

## Relational analysis of IS_B2_A1_B2_A2_A2_A2

### Relational analysis result of IS_B2_A1_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0391442, upper bound: 0.0393559
time: 2.53 seconds

## BFS IS instance: IS_B2_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0344545, 0.0307241, -0.0398527, 0.0331742, -0.0676287, 0.0705768
1: -0.0254072, 0.0257039, -0.0295302, 0.0274830, -0.0528901, 0.0552341
2: -0.0612442, 0.0172353, -0.0677077, 0.0196906, -0.0809347, 0.0849430
3: 1.0026433, 1.0281451, 1.0017530, 1.0295537, -0.0269103, 0.0263921
4: -0.0113453, 0.0403924, -0.0131895, 0.0449146, -0.0562600, 0.0535819
5: -0.0135740, 0.0789206, -0.0161695, 0.0855209, -0.0990949, 0.0950900
6: -0.0597351, 0.0281679, -0.0645385, 0.0333658, -0.0931009, 0.0927063
7: -0.0487449, -0.0016871, -0.0522432, -0.0016347, -0.0471007, 0.0504741
8: -0.0274699, 0.0260641, -0.0303714, 0.0302927, -0.0577626, 0.0564354
9: -0.0270159, 0.0438389, -0.0291499, 0.0472414, -0.0742573, 0.0729888

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of IS_B2_A2_A1_B2_B1_A1

### Relational analysis result of IS_B2_A2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392687, upper bound: 0.0391755
time: 2.47 seconds

## Relational analysis of IS_B2_A2_A1_B2_B1_A2

### Relational analysis result of IS_B2_A2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0390093, upper bound: 0.0391710
time: 2.07 seconds

## BFS IS instance: IS_B2_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0326467, 0.0301799, -0.0442761, 0.0350194, -0.0676662, 0.0744560
1: -0.0240534, 0.0250694, -0.0322787, 0.0291041, -0.0531575, 0.0573481
2: -0.0589612, 0.0172038, -0.0726809, 0.0218582, -0.0808194, 0.0898847
3: 1.0028874, 1.0275606, 1.0009059, 1.0310975, -0.0282102, 0.0266547
4: -0.0107216, 0.0388874, -0.0143785, 0.0491666, -0.0598882, 0.0532659
5: -0.0129079, 0.0767298, -0.0189593, 0.0900486, -0.1029565, 0.0956892
6: -0.0580864, 0.0263558, -0.0683652, 0.0379492, -0.0960356, 0.0947210
7: -0.0478585, -0.0017006, -0.0554434, -0.0012244, -0.0466341, 0.0536604
8: -0.0265902, 0.0245963, -0.0327541, 0.0337819, -0.0603721, 0.0573503
9: -0.0262792, 0.0426792, -0.0312476, 0.0497081, -0.0759873, 0.0739268

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=10, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 207

## Relational analysis of IS_B2_A2_A1_B2_B2_B1

### Relational analysis result of IS_B2_A2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0391416, upper bound: 0.0395897
time: 2.29 seconds

## Relational analysis of IS_B2_A2_A1_B2_B2_B2

### Relational analysis result of IS_B2_A2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0391413, upper bound: 0.0394414
time: 2.48 seconds

## BFS IS instance: IS_B2_A2_A2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0172603, 0.0242716, -0.0414090, 0.0339671, -0.0512274, 0.0656806
1: -0.0135061, 0.0179203, -0.0306579, 0.0282450, -0.0417511, 0.0485782
2: -0.0363242, 0.0168712, -0.0699176, 0.0203902, -0.0567145, 0.0867888
3: 1.0040301, 1.0207944, 1.0015322, 1.0303609, -0.0263308, 0.0192622
4: -0.0043084, 0.0256633, -0.0136748, 0.0463711, -0.0506795, 0.0393381
5: -0.0052660, 0.0590275, -0.0172985, 0.0873797, -0.0926457, 0.0763260
6: -0.0416389, 0.0073078, -0.0662613, 0.0349608, -0.0765997, 0.0735691
7: -0.0383574, -0.0017341, -0.0533440, -0.0016475, -0.0367029, 0.0515376
8: -0.0196752, 0.0094781, -0.0314757, 0.0315632, -0.0512384, 0.0409538
9: -0.0198390, 0.0303846, -0.0298703, 0.0482852, -0.0681241, 0.0602549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 207

## Relational analysis of IS_B2_A2_A2_A1_A1_B1

### Relational analysis result of IS_B2_A2_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393301, upper bound: 0.0393989
time: 2.41 seconds

## Relational analysis of IS_B2_A2_A2_A1_A1_B2

### Relational analysis result of IS_B2_A2_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393301, upper bound: 0.0393989
time: 2.58 seconds

## BFS IS instance: IS_B2_A2_A2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0304371, 0.0292791, -0.0518934, 0.0415928, -0.0720298, 0.0811725
1: -0.0225668, 0.0239135, -0.0365512, 0.0317193, -0.0542861, 0.0604648
2: -0.0555348, 0.0171588, -0.0803235, 0.0278234, -0.0833582, 0.0974823
3: 1.0030581, 1.0264103, 0.9990472, 1.0329884, -0.0299304, 0.0273631
4: -0.0098535, 0.0368791, -0.0178166, 0.0572695, -0.0671230, 0.0546957
5: -0.0116343, 0.0741545, -0.0246675, 0.0970326, -0.1086669, 0.0988221
6: -0.0555302, 0.0236404, -0.0750475, 0.0469345, -0.1024647, 0.0986879
7: -0.0464418, -0.0017396, -0.0612848, -0.0014020, -0.0450399, 0.0594744
8: -0.0254606, 0.0224053, -0.0361195, 0.0427187, -0.0681793, 0.0585249
9: -0.0252930, 0.0408781, -0.0363953, 0.0541175, -0.0794105, 0.0772735

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 207

## Relational analysis of IS_B2_A2_A2_A1_A2_B1

### Relational analysis result of IS_B2_A2_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0397541, upper bound: 0.0395347
time: 2.84 seconds

## Relational analysis of IS_B2_A2_A2_A1_A2_B2

### Relational analysis result of IS_B2_A2_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0397541, upper bound: 0.0395347
time: 2.37 seconds

## BFS IS instance: IS_B2_A2_A2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0360191, 0.0318601, -0.0561572, 0.0458352, -0.0818543, 0.0880173
1: -0.0261617, 0.0277531, -0.0389444, 0.0332543, -0.0594161, 0.0666975
2: -0.0654054, 0.0171909, -0.0846751, 0.0313676, -0.0967730, 0.1018660
3: 1.0027057, 1.0307668, 0.9978117, 1.0340009, -0.0312952, 0.0329551
4: -0.0114785, 0.0428332, -0.0198965, 0.0619702, -0.0734487, 0.0627297
5: -0.0162217, 0.0810479, -0.0279548, 0.1009864, -0.1172081, 0.1090026
6: -0.0633906, 0.0299733, -0.0788864, 0.0521609, -0.1155515, 0.1088597
7: -0.0502736, -0.0020075, -0.0645941, -0.0012643, -0.0490093, 0.0625548
8: -0.0295473, 0.0274748, -0.0379677, 0.0481673, -0.0777145, 0.0654425
9: -0.0282911, 0.0455488, -0.0394197, 0.0567664, -0.0850575, 0.0849684

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=28, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B2_A2_A2_A2_A1_B1

### Relational analysis result of IS_B2_A2_A2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0365297, upper bound: 0.0361169
time: 6.63 seconds

## Relational analysis of IS_B2_A2_A2_A2_A1_B2

### Relational analysis result of IS_B2_A2_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0383102, upper bound: 0.0390702
time: 2.45 seconds

## BFS IS instance: IS_B2_A2_A2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0799659, 0.0482325, -0.0561572, 0.0458352, -0.1258011, 0.1043898
1: -0.0522660, 0.0403815, -0.0389444, 0.0332543, -0.0855203, 0.0793259
2: -0.1079613, 0.0399752, -0.0846751, 0.0313676, -0.1393289, 0.1246502
3: 0.9924181, 1.0413095, 0.9978117, 1.0340009, -0.0415828, 0.0434977
4: -0.0230859, 0.0842906, -0.0198965, 0.0619702, -0.0850561, 0.1041872
5: -0.0407492, 0.1229124, -0.0279548, 0.1009864, -0.1417357, 0.1508672
6: -0.0953666, 0.0755647, -0.0788864, 0.0521609, -0.1475275, 0.1544510
7: -0.0818736, 0.0031549, -0.0645941, -0.0012643, -0.0806093, 0.0677490
8: -0.0493473, 0.0617792, -0.0379677, 0.0481673, -0.0975146, 0.0997470
9: -0.0487387, 0.0674511, -0.0394197, 0.0567664, -0.1055051, 0.1068708

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_B2_A2_A2_A2_A2_A1

### Relational analysis result of IS_B2_A2_A2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0378977, upper bound: 0.0393483
time: 2.60 seconds

## Relational analysis of IS_B2_A2_A2_A2_A2_A2

### Relational analysis result of IS_B2_A2_A2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0381837, upper bound: 0.0395085
time: 2.89 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 6.82 seconds
IS_B1_B1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 3, lower bound: -0.0383680, upper bound: 0.0380875
IS_B1_B1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 3, lower bound: -0.0383680, upper bound: 0.0380875
IS_B1_B1_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 3, lower bound: -0.0383680, upper bound: 0.0380875
IS_B1_B1_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 3, lower bound: -0.0383680, upper bound: 0.0380875
IS_B1_B1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 3, lower bound: -0.0383630, upper bound: 0.0380873
IS_B1_B1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 3, lower bound: -0.0383630, upper bound: 0.0380873
IS_B1_B1_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 3, lower bound: -0.0383630, upper bound: 0.0380873
IS_B1_B1_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 3, lower bound: -0.0383630, upper bound: 0.0380873
IS_B1_B1_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 3, lower bound: -0.0383847, upper bound: 0.0381755
IS_B1_B1_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 3, lower bound: -0.0383847, upper bound: 0.0381754
IS_B1_B1_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 3, lower bound: -0.0383900, upper bound: 0.0381924
IS_B1_B1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 3, lower bound: -0.0383900, upper bound: 0.0381925
IS_B1_B2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 3, lower bound: -0.0389513, upper bound: 0.0386831
IS_B1_B2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 3, lower bound: -0.0389513, upper bound: 0.0389339
IS_B1_B2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 3, lower bound: -0.0389513, upper bound: 0.0386269
IS_B1_B2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 3, lower bound: -0.0389513, upper bound: 0.0389339
IS_B1_B2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 3, lower bound: -0.0389567, upper bound: 0.0386862
IS_B1_B2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 3, lower bound: -0.0389567, upper bound: 0.0389249
IS_B1_B2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 3, lower bound: -0.0389567, upper bound: 0.0386862
IS_B1_B2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 3, lower bound: -0.0389567, upper bound: 0.0389249
IS_B1_B2_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 3, lower bound: -0.0386284, upper bound: 0.0380903
IS_B1_B2_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 3, lower bound: -0.0386284, upper bound: 0.0382473
IS_B1_B2_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 3, lower bound: -0.0386354, upper bound: 0.0380909
IS_B1_B2_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 3, lower bound: -0.0386354, upper bound: 0.0382221
IS_B1_B2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 3, lower bound: -0.0386315, upper bound: 0.0381076
IS_B1_B2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 3, lower bound: -0.0386315, upper bound: 0.0382408
IS_B1_B2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 3, lower bound: -0.0386382, upper bound: 0.0381128
IS_B1_B2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 3, lower bound: -0.0386382, upper bound: 0.0382406
IS_B2_A1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 3, lower bound: -0.0380114, upper bound: 0.0387253
IS_B2_A1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 3, lower bound: -0.0380114, upper bound: 0.0394151
IS_B2_A1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 3, lower bound: -0.0380114, upper bound: 0.0387253
IS_B2_A1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 3, lower bound: -0.0380114, upper bound: 0.0394151
IS_B2_A1_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 3, lower bound: -0.0391260, upper bound: 0.0390701
IS_B2_A1_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 3, lower bound: -0.0391260, upper bound: 0.0390486
IS_B2_A1_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 3, lower bound: -0.0394573, upper bound: 0.0393771
IS_B2_A1_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 3, lower bound: -0.0391354, upper bound: 0.0393532
IS_B2_A1_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 3, lower bound: -0.0394584, upper bound: 0.0393807
IS_B2_A1_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 3, lower bound: -0.0391442, upper bound: 0.0393559
IS_B2_A2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 3, lower bound: -0.0392687, upper bound: 0.0391755
IS_B2_A2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 3, lower bound: -0.0390093, upper bound: 0.0391710
IS_B2_A2_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 3, lower bound: -0.0391416, upper bound: 0.0395897
IS_B2_A2_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 3, lower bound: -0.0391413, upper bound: 0.0394414
IS_B2_A2_A2_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 3, lower bound: -0.0393301, upper bound: 0.0393989
IS_B2_A2_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 3, lower bound: -0.0393301, upper bound: 0.0393989
IS_B2_A2_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 3, lower bound: -0.0397541, upper bound: 0.0395347
IS_B2_A2_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 3, lower bound: -0.0397541, upper bound: 0.0395347
IS_B2_A2_A2_A2_A1_B1, status: Status.VERIFIED, split count: 6, time: 6.82
Output dim: 3, lower bound: -0.0365297, upper bound: 0.0361169
IS_B2_A2_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 3, lower bound: -0.0383102, upper bound: 0.0390702
IS_B2_A2_A2_A2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 3, lower bound: -0.0378977, upper bound: 0.0393483
IS_B2_A2_A2_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 3, lower bound: -0.0381837, upper bound: 0.0395085

## BFS IS instance: IS_B1_B1_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0180227, 0.0241764, -0.0064515, 0.0125963, -0.0306191, 0.0302603
1: -0.0142838, 0.0172305, -0.0050600, 0.0050447, -0.0193285, 0.0222905
2: -0.0360017, 0.0169461, 0.0000001, 0.0161670, -0.0521687, 0.0169460
3: 1.0038850, 1.0195788, 1.0063828, 1.0119085, -0.0080235, 0.0131960
4: -0.0050164, 0.0255386, -0.0041914, 0.0005900, -0.0056064, 0.0297300
5: -0.0042598, 0.0597600, -0.0010086, 0.0198185, -0.0240783, 0.0605656
6: -0.0410256, 0.0082647, -0.0152119, 0.0001812, -0.0412068, 0.0234766
7: -0.0384253, -0.0018648, -0.0211053, -0.0019382, -0.0364487, 0.0192034
8: -0.0189425, 0.0101992, -0.0125256, 0.0034622, -0.0224048, 0.0227248
9: -0.0197056, 0.0307044, -0.0099261, 0.0031540, -0.0228595, 0.0406305

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=25, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 229

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B1_B1_B1_A1_A1_B1_A1

### Relational analysis result of IS_B1_B1_B1_A1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0356515, upper bound: 0.0360434
time: 2.09 seconds

## Relational analysis of IS_B1_B1_B1_A1_A1_B1_A2

### Relational analysis result of IS_B1_B1_B1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0385089, upper bound: 0.0380754
time: 2.72 seconds

## BFS IS instance: IS_B1_B1_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0180227, 0.0241764, -0.0066633, 0.0101810, -0.0282038, 0.0304816
1: -0.0142838, 0.0172305, -0.0050584, 0.0025731, -0.0168569, 0.0222889
2: -0.0360017, 0.0169461, 0.0041514, 0.0160263, -0.0520281, 0.0127947
3: 1.0038850, 1.0195788, 1.0064046, 1.0107716, -0.0068866, 0.0131742
4: -0.0050164, 0.0255386, -0.0041744, -0.0006056, -0.0044108, 0.0297131
5: -0.0042598, 0.0597600, -0.0011445, 0.0166135, -0.0208733, 0.0606136
6: -0.0410256, 0.0082647, -0.0125753, -0.0013007, -0.0397249, 0.0208400
7: -0.0384253, -0.0018648, -0.0199107, -0.0013224, -0.0370652, 0.0180270
8: -0.0189425, 0.0101992, -0.0123830, 0.0033396, -0.0222821, 0.0225822
9: -0.0197056, 0.0307044, -0.0079688, 0.0022279, -0.0219335, 0.0386732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=25, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 207

## Relational analysis of IS_B1_B1_B1_A1_A1_B2_A1

### Relational analysis result of IS_B1_B1_B1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0385614, upper bound: 0.0381410
time: 3.50 seconds

## Relational analysis of IS_B1_B1_B1_A1_A1_B2_A2

### Relational analysis result of IS_B1_B1_B1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0385614, upper bound: 0.0381410
time: 2.19 seconds

## BFS IS instance: IS_B1_B1_B1_A1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0116004, 0.0201576, -0.0064597, 0.0138134, -0.0254138, 0.0264469
1: -0.0107580, 0.0122194, -0.0054086, 0.0063918, -0.0171499, 0.0176280
2: -0.0231734, 0.0167650, -0.0041699, 0.0162354, -0.0394088, 0.0209349
3: 1.0045333, 1.0154262, 1.0061179, 1.0127951, -0.0082618, 0.0093082
4: -0.0042868, 0.0169687, -0.0042000, 0.0034130, -0.0076999, 0.0211687
5: -0.0015403, 0.0479853, -0.0010405, 0.0245996, -0.0261399, 0.0490258
6: -0.0306516, 0.0039451, -0.0178121, 0.0010739, -0.0317256, 0.0217572
7: -0.0326222, -0.0015870, -0.0231167, -0.0019355, -0.0306519, 0.0214979
8: -0.0148608, 0.0065471, -0.0126776, 0.0038719, -0.0187327, 0.0192248
9: -0.0161286, 0.0225301, -0.0110309, 0.0065122, -0.0226408, 0.0335610

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of IS_B1_B1_B1_A1_A2_A1_B1

### Relational analysis result of IS_B1_B1_B1_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0383680, upper bound: 0.0380875
time: 2.39 seconds

## Relational analysis of IS_B1_B1_B1_A1_A2_A1_B2

### Relational analysis result of IS_B1_B1_B1_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0383680, upper bound: 0.0380875
time: 2.68 seconds

## BFS IS instance: IS_B1_B1_B1_A1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0296420, 0.0329564, -0.0064597, 0.0138134, -0.0434554, 0.0392336
1: -0.0199221, 0.0280948, -0.0054086, 0.0063918, -0.0263140, 0.0335033
2: -0.0623240, 0.0174072, -0.0041699, 0.0162354, -0.0785594, 0.0215771
3: 1.0022479, 1.0285637, 1.0061179, 1.0127951, -0.0105472, 0.0224458
4: -0.0043760, 0.0442745, -0.0042000, 0.0034130, -0.0077890, 0.0484746
5: -0.0118147, 0.0856398, -0.0010405, 0.0245996, -0.0364143, 0.0866803
6: -0.0636402, 0.0121708, -0.0178121, 0.0010739, -0.0647141, 0.0299829
7: -0.0506986, -0.0016265, -0.0231167, -0.0019355, -0.0487257, 0.0214536
8: -0.0280145, 0.0155307, -0.0126776, 0.0038719, -0.0318865, 0.0282084
9: -0.0272177, 0.0487683, -0.0110309, 0.0065122, -0.0337299, 0.0597991

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B1_B1_B1_A1_A2_A2_A1

### Relational analysis result of IS_B1_B1_B1_A1_A2_A2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0353159, upper bound: 0.0358989
time: 1.68 seconds

## Relational analysis of IS_B1_B1_B1_A1_A2_A2_A2

### Relational analysis result of IS_B1_B1_B1_A1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0383112, upper bound: 0.0380213
time: 2.15 seconds

## BFS IS instance: IS_B1_B1_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0214768, 0.0255405, -0.0064494, 0.0122694, -0.0337462, 0.0316435
1: -0.0166079, 0.0189620, -0.0050552, 0.0047240, -0.0213319, 0.0240172
2: -0.0412403, 0.0170174, 0.0005961, 0.0161426, -0.0573829, 0.0164213
3: 1.0036145, 1.0212578, 1.0064076, 1.0117506, -0.0081360, 0.0148501
4: -0.0064272, 0.0285483, -0.0041879, 0.0003657, -0.0067930, 0.0327362
5: -0.0060155, 0.0637137, -0.0010030, 0.0192876, -0.0253032, 0.0644839
6: -0.0447880, 0.0125806, -0.0148170, 0.0000036, -0.0447917, 0.0273976
7: -0.0405940, -0.0015429, -0.0209080, -0.0019419, -0.0386165, 0.0193310
8: -0.0205802, 0.0136091, -0.0124875, 0.0033820, -0.0239622, 0.0260966
9: -0.0211712, 0.0334895, -0.0096481, 0.0029431, -0.0241143, 0.0431376

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=25, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 207

## Relational analysis of IS_B1_B1_B1_A2_A1_B1_A1

### Relational analysis result of IS_B1_B1_B1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0385585, upper bound: 0.0381408
time: 2.46 seconds

## Relational analysis of IS_B1_B1_B1_A2_A1_B1_A2

### Relational analysis result of IS_B1_B1_B1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0385585, upper bound: 0.0381408
time: 2.58 seconds

## BFS IS instance: IS_B1_B1_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0214768, 0.0255405, -0.0066616, 0.0098798, -0.0313566, 0.0318678
1: -0.0166079, 0.0189620, -0.0050580, 0.0022867, -0.0188945, 0.0240200
2: -0.0412403, 0.0170174, 0.0045850, 0.0160039, -0.0572441, 0.0124324
3: 1.0036145, 1.0212578, 1.0064205, 1.0106637, -0.0070492, 0.0148373
4: -0.0064272, 0.0285483, -0.0041712, -0.0007364, -0.0056908, 0.0327195
5: -0.0060155, 0.0637137, -0.0011417, 0.0162289, -0.0222444, 0.0645777
6: -0.0447880, 0.0125806, -0.0122530, -0.0014502, -0.0433378, 0.0248336
7: -0.0405940, -0.0015429, -0.0197776, -0.0013262, -0.0392329, 0.0182179
8: -0.0205802, 0.0136091, -0.0123527, 0.0033107, -0.0238908, 0.0259618
9: -0.0211712, 0.0334895, -0.0077531, 0.0021360, -0.0233072, 0.0412426

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=25, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 207

## Relational analysis of IS_B1_B1_B1_A2_A1_B2_A1

### Relational analysis result of IS_B1_B1_B1_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0385585, upper bound: 0.0381408
time: 2.27 seconds

## Relational analysis of IS_B1_B1_B1_A2_A1_B2_A2

### Relational analysis result of IS_B1_B1_B1_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0385585, upper bound: 0.0381408
time: 2.36 seconds

## BFS IS instance: IS_B1_B1_B1_A2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0135597, 0.0215585, -0.0064558, 0.0134927, -0.0270525, 0.0277858
1: -0.0117374, 0.0140149, -0.0052770, 0.0060863, -0.0178237, 0.0192919
2: -0.0275201, 0.0168336, -0.0031583, 0.0162105, -0.0437305, 0.0199919
3: 1.0042560, 1.0169663, 1.0061878, 1.0126445, -0.0083885, 0.0107785
4: -0.0042965, 0.0199485, -0.0041965, 0.0026740, -0.0069706, 0.0241450
5: -0.0022588, 0.0520126, -0.0010333, 0.0233215, -0.0255803, 0.0530459
6: -0.0342891, 0.0049148, -0.0171212, 0.0009156, -0.0352047, 0.0220359
7: -0.0345967, -0.0012550, -0.0226012, -0.0019392, -0.0326238, 0.0213172
8: -0.0163660, 0.0074696, -0.0126224, 0.0037702, -0.0201363, 0.0200920
9: -0.0173574, 0.0253622, -0.0107575, 0.0056380, -0.0229953, 0.0361197

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 134

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of IS_B1_B1_B1_A2_A2_A1_B1

### Relational analysis result of IS_B1_B1_B1_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0383630, upper bound: 0.0380873
time: 2.28 seconds

## Relational analysis of IS_B1_B1_B1_A2_A2_A1_B2

### Relational analysis result of IS_B1_B1_B1_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0383630, upper bound: 0.0380873
time: 2.14 seconds

## BFS IS instance: IS_B1_B1_B1_A2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0312630, 0.0341000, -0.0064558, 0.0134927, -0.0447557, 0.0403109
1: -0.0207523, 0.0294928, -0.0052770, 0.0060863, -0.0268386, 0.0347698
2: -0.0658014, 0.0174695, -0.0031583, 0.0162105, -0.0820119, 0.0206278
3: 1.0020145, 1.0297178, 1.0061878, 1.0126445, -0.0106300, 0.0235300
4: -0.0043842, 0.0467186, -0.0041965, 0.0026740, -0.0070583, 0.0509151
5: -0.0127877, 0.0890444, -0.0010333, 0.0233215, -0.0361092, 0.0900777
6: -0.0665694, 0.0128808, -0.0171212, 0.0009156, -0.0674850, 0.0300020
7: -0.0523154, -0.0013205, -0.0226012, -0.0019392, -0.0503403, 0.0212491
8: -0.0291751, 0.0164151, -0.0126224, 0.0037702, -0.0329453, 0.0290375
9: -0.0282051, 0.0511386, -0.0107575, 0.0056380, -0.0338430, 0.0618960

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B1_B1_B1_A2_A2_A2_A1

### Relational analysis result of IS_B1_B1_B1_A2_A2_A2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0352890, upper bound: 0.0358887
time: 1.98 seconds

## Relational analysis of IS_B1_B1_B1_A2_A2_A2_A2

### Relational analysis result of IS_B1_B1_B1_A2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0383059, upper bound: 0.0380213
time: 2.16 seconds

## BFS IS instance: IS_B1_B1_B2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0155873, 0.0231546, -0.0218776, 0.0276241, -0.0432114, 0.0450322
1: -0.0125818, 0.0167662, -0.0157712, 0.0223366, -0.0349184, 0.0325374
2: -0.0330669, 0.0167786, -0.0467453, 0.0169898, -0.0500567, 0.0635240
3: 1.0043164, 1.0200183, 1.0035412, 1.0246814, -0.0203650, 0.0164771
4: -0.0042769, 0.0231994, -0.0043045, 0.0327272, -0.0370041, 0.0275040
5: -0.0045241, 0.0553105, -0.0084543, 0.0684115, -0.0729356, 0.0637648
6: -0.0388209, 0.0068017, -0.0503454, 0.0097007, -0.0485216, 0.0571472
7: -0.0367349, -0.0020629, -0.0430303, -0.0019405, -0.0347368, 0.0409008
8: -0.0186992, 0.0086946, -0.0232956, 0.0119794, -0.0306786, 0.0319901
9: -0.0188839, 0.0278387, -0.0227635, 0.0369783, -0.0558621, 0.0506022

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=24, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 207

## Relational analysis of IS_B1_B1_B2_B2_B1_A1_A1

### Relational analysis result of IS_B1_B1_B2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0383847, upper bound: 0.0381755
time: 2.49 seconds

## Relational analysis of IS_B1_B1_B2_B2_B1_A1_A2

### Relational analysis result of IS_B1_B1_B2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0383847, upper bound: 0.0381755
time: 2.82 seconds

## BFS IS instance: IS_B1_B1_B2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0457863, 0.0360236, -0.0218776, 0.0276241, -0.0734104, 0.0579012
1: -0.0340079, 0.0296832, -0.0157712, 0.0223366, -0.0563445, 0.0454544
2: -0.0751298, 0.0225905, -0.0467453, 0.0169898, -0.0921196, 0.0693359
3: 1.0007987, 1.0314904, 1.0035412, 1.0246814, -0.0238827, 0.0279492
4: -0.0151744, 0.0500366, -0.0043045, 0.0327272, -0.0479016, 0.0543411
5: -0.0194684, 0.0927183, -0.0084543, 0.0684115, -0.0878799, 0.1011726
6: -0.0701478, 0.0391785, -0.0503454, 0.0097007, -0.0798486, 0.0895239
7: -0.0562499, -0.0016421, -0.0430303, -0.0019405, -0.0542617, 0.0413882
8: -0.0338638, 0.0349901, -0.0232956, 0.0119794, -0.0458433, 0.0582857
9: -0.0316026, 0.0510328, -0.0227635, 0.0369783, -0.0685809, 0.0737963

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=24, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=7, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of IS_B1_B1_B2_B2_B1_A2_A1

### Relational analysis result of IS_B1_B1_B2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0380217, upper bound: 0.0378374
time: 2.15 seconds

## Relational analysis of IS_B1_B1_B2_B2_B1_A2_A2

### Relational analysis result of IS_B1_B1_B2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0380480, upper bound: 0.0378374
time: 2.31 seconds

## BFS IS instance: IS_B1_B1_B2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0149536, 0.0227102, -0.0235690, 0.0288303, -0.0437839, 0.0462792
1: -0.0122554, 0.0162338, -0.0166220, 0.0238704, -0.0361259, 0.0328558
2: -0.0317193, 0.0167512, -0.0504677, 0.0170506, -0.0487698, 0.0672189
3: 1.0043988, 1.0195923, 1.0032989, 1.0260026, -0.0216038, 0.0162934
4: -0.0042730, 0.0222461, -0.0043129, 0.0352911, -0.0395641, 0.0265589
5: -0.0041690, 0.0539669, -0.0095685, 0.0718954, -0.0760644, 0.0635354
6: -0.0376849, 0.0065367, -0.0534687, 0.0105209, -0.0482058, 0.0600053
7: -0.0361019, -0.0020658, -0.0447233, -0.0015685, -0.0344796, 0.0425949
8: -0.0182591, 0.0083946, -0.0245937, 0.0129158, -0.0311750, 0.0329883
9: -0.0185030, 0.0268978, -0.0238194, 0.0394335, -0.0579366, 0.0507172

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=24, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 207

## Relational analysis of IS_B1_B1_B2_B2_B2_A1_A1

### Relational analysis result of IS_B1_B1_B2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0383900, upper bound: 0.0381925
time: 2.10 seconds

## Relational analysis of IS_B1_B1_B2_B2_B2_A1_A2

### Relational analysis result of IS_B1_B1_B2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0383900, upper bound: 0.0381925
time: 2.23 seconds

## BFS IS instance: IS_B1_B1_B2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0441894, 0.0352524, -0.0235690, 0.0288303, -0.0730196, 0.0588214
1: -0.0328093, 0.0290735, -0.0166220, 0.0238704, -0.0566798, 0.0456955
2: -0.0731027, 0.0218253, -0.0504677, 0.0170506, -0.0901532, 0.0722930
3: 1.0010480, 1.0309441, 1.0032989, 1.0260026, -0.0249547, 0.0276452
4: -0.0146439, 0.0486448, -0.0043129, 0.0352911, -0.0499350, 0.0529577
5: -0.0185600, 0.0907787, -0.0095685, 0.0718954, -0.0904554, 0.1003472
6: -0.0686090, 0.0376075, -0.0534687, 0.0105209, -0.0791299, 0.0910762
7: -0.0551630, -0.0016888, -0.0447233, -0.0015685, -0.0535506, 0.0430346
8: -0.0329136, 0.0337190, -0.0245937, 0.0129158, -0.0458294, 0.0583127
9: -0.0309347, 0.0499978, -0.0238194, 0.0394335, -0.0703682, 0.0738173

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=24, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=7, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of IS_B1_B1_B2_B2_B2_A2_A1

### Relational analysis result of IS_B1_B1_B2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0380270, upper bound: 0.0378584
time: 2.48 seconds

## Relational analysis of IS_B1_B1_B2_B2_B2_A2_A2

### Relational analysis result of IS_B1_B1_B2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0380546, upper bound: 0.0378584
time: 2.07 seconds

## BFS IS instance: IS_B1_B2_B1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0228301, 0.0263464, -0.0109940, 0.0198974, -0.0427274, 0.0373404
1: -0.0173641, 0.0203311, -0.0102430, 0.0127564, -0.0301205, 0.0305740
2: -0.0442627, 0.0170042, -0.0231229, 0.0166022, -0.0608648, 0.0401271
3: 1.0035926, 1.0229542, 1.0049424, 1.0167452, -0.0131526, 0.0180118
4: -0.0067005, 0.0303147, -0.0042506, 0.0162394, -0.0229399, 0.0345653
5: -0.0077518, 0.0654179, -0.0019434, 0.0456744, -0.0534262, 0.0673613
6: -0.0473363, 0.0141940, -0.0304295, 0.0047489, -0.0520853, 0.0446235
7: -0.0417259, -0.0016938, -0.0321205, -0.0023162, -0.0393645, 0.0303759
8: -0.0219863, 0.0148530, -0.0153606, 0.0065203, -0.0285066, 0.0302137
9: -0.0220879, 0.0347879, -0.0160696, 0.0211202, -0.0432081, 0.0508575

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=25, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_B1_B2_B1_B1_A1_A1_A1

### Relational analysis result of IS_B1_B2_B1_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0389626, upper bound: 0.0391095
time: 4.53 seconds

## Relational analysis of IS_B1_B2_B1_B1_A1_A1_A2

### Relational analysis result of IS_B1_B2_B1_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0389626, upper bound: 0.0391095
time: 2.27 seconds

## BFS IS instance: IS_B1_B2_B1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0343849, 0.0306071, -0.0109940, 0.0198974, -0.0542823, 0.0416011
1: -0.0254185, 0.0254305, -0.0102430, 0.0127564, -0.0381748, 0.0356735
2: -0.0607846, 0.0172440, -0.0231229, 0.0166022, -0.0773867, 0.0403669
3: 1.0026504, 1.0277591, 1.0049424, 1.0167452, -0.0140948, 0.0228167
4: -0.0113749, 0.0401613, -0.0042506, 0.0162394, -0.0276143, 0.0444119
5: -0.0131772, 0.0788518, -0.0019434, 0.0456744, -0.0588516, 0.0807951
6: -0.0593045, 0.0280173, -0.0304295, 0.0047489, -0.0640535, 0.0584468
7: -0.0485852, -0.0020680, -0.0321205, -0.0023162, -0.0462169, 0.0300104
8: -0.0271832, 0.0259646, -0.0153606, 0.0065203, -0.0337035, 0.0413252
9: -0.0268751, 0.0437074, -0.0160696, 0.0211202, -0.0479953, 0.0597770

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=25, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_B1_B2_B1_B1_A1_A2_A1

### Relational analysis result of IS_B1_B2_B1_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0389626, upper bound: 0.0391864
time: 2.48 seconds

## Relational analysis of IS_B1_B2_B1_B1_A1_A2_A2

### Relational analysis result of IS_B1_B2_B1_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0389626, upper bound: 0.0391864
time: 2.43 seconds

## BFS IS instance: IS_B1_B2_B1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0696013, 0.0597221, -0.0109940, 0.0198974, -0.0894987, 0.0707161
1: -0.0464658, 0.0392985, -0.0102430, 0.0127564, -0.0592222, 0.0495415
2: -0.1003697, 0.0413380, -0.0231229, 0.0166022, -0.1169719, 0.0644609
3: 0.9941134, 1.0384228, 1.0049424, 1.0167452, -0.0226318, 0.0334804
4: -0.0257646, 0.0778540, -0.0042506, 0.0162394, -0.0420040, 0.0821046
5: -0.0388279, 0.1140628, -0.0019434, 0.0456744, -0.0845023, 0.1160062
6: -0.0919989, 0.0694045, -0.0304295, 0.0047489, -0.0967478, 0.0998340
7: -0.0753544, 0.0008712, -0.0321205, -0.0023162, -0.0729838, 0.0329917
8: -0.0445661, 0.0655786, -0.0153606, 0.0065203, -0.0510864, 0.0809393
9: -0.0487116, 0.0663204, -0.0160696, 0.0211202, -0.0698318, 0.0823900

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=25, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of IS_B1_B2_B1_B1_A2_A1_A1

### Relational analysis result of IS_B1_B2_B1_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0384533, upper bound: 0.0381372
time: 2.98 seconds

## Relational analysis of IS_B1_B2_B1_B1_A2_A1_A2

### Relational analysis result of IS_B1_B2_B1_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0382680, upper bound: 0.0380991
time: 3.07 seconds

## BFS IS instance: IS_B1_B2_B1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0889612, 0.0791658, -0.0109940, 0.0198974, -0.1088585, 0.0901597
1: -0.0573312, 0.0466859, -0.0102430, 0.0127564, -0.0700875, 0.0569288
2: -0.1208464, 0.0569573, -0.0231229, 0.0166022, -0.1374486, 0.0800802
3: 0.9885534, 1.0434369, 1.0049424, 1.0167452, -0.0281918, 0.0384945
4: -0.0349440, 0.0995761, -0.0042506, 0.0162394, -0.0511833, 0.1038267
5: -0.0539728, 0.1322747, -0.0019434, 0.0456744, -0.0996472, 0.1342181
6: -0.1097840, 0.0933835, -0.0304295, 0.0047489, -0.1145330, 0.1238130
7: -0.0905114, 0.0053534, -0.0321205, -0.0023162, -0.0881364, 0.0374739
8: -0.0532186, 0.0904440, -0.0153606, 0.0065203, -0.0597389, 0.1058047
9: -0.0623727, 0.0788254, -0.0160696, 0.0211202, -0.0834928, 0.0948951

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=25, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B1_B2_B1_B1_A2_A2_A1

### Relational analysis result of IS_B1_B2_B1_B1_A2_A2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0359013, upper bound: 0.0370929
time: 2.22 seconds

## Relational analysis of IS_B1_B2_B1_B1_A2_A2_A2

### Relational analysis result of IS_B1_B2_B1_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0389022, upper bound: 0.0388818
time: 3.11 seconds

## BFS IS instance: IS_B1_B2_B1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0216227, 0.0259012, -0.0127249, 0.0211417, -0.0427643, 0.0386261
1: -0.0165244, 0.0198247, -0.0111069, 0.0143605, -0.0308849, 0.0309316
2: -0.0425603, 0.0169788, -0.0269850, 0.0166598, -0.0592200, 0.0439638
3: 1.0036784, 1.0224959, 1.0046966, 1.0181364, -0.0144579, 0.0177993
4: -0.0061818, 0.0293138, -0.0042589, 0.0188851, -0.0250669, 0.0335727
5: -0.0072231, 0.0640255, -0.0029642, 0.0492277, -0.0564508, 0.0669896
6: -0.0461195, 0.0127054, -0.0336692, 0.0056171, -0.0517366, 0.0463746
7: -0.0409968, -0.0016976, -0.0338720, -0.0019661, -0.0389874, 0.0321266
8: -0.0215127, 0.0136562, -0.0167138, 0.0073999, -0.0289126, 0.0303700
9: -0.0216054, 0.0338325, -0.0171620, 0.0236212, -0.0452266, 0.0509945

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=25, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_B1_B2_B1_B2_A1_A1_A1

### Relational analysis result of IS_B1_B2_B1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0389630, upper bound: 0.0390924
time: 2.46 seconds

## Relational analysis of IS_B1_B2_B1_B2_A1_A1_A2

### Relational analysis result of IS_B1_B2_B1_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0389630, upper bound: 0.0390924
time: 2.46 seconds

## BFS IS instance: IS_B1_B2_B1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0326466, 0.0301007, -0.0127249, 0.0211417, -0.0537883, 0.0428256
1: -0.0241018, 0.0248473, -0.0111069, 0.0143605, -0.0384623, 0.0359542
2: -0.0586406, 0.0172132, -0.0269850, 0.0166598, -0.0753003, 0.0441982
3: 1.0028995, 1.0272306, 1.0046966, 1.0181364, -0.0152369, 0.0225340
4: -0.0107919, 0.0387197, -0.0042589, 0.0188851, -0.0296770, 0.0429786
5: -0.0125756, 0.0767265, -0.0029642, 0.0492277, -0.0618033, 0.0796906
6: -0.0577572, 0.0263316, -0.0336692, 0.0056171, -0.0633743, 0.0600008
7: -0.0477632, -0.0020832, -0.0338720, -0.0019661, -0.0457486, 0.0317436
8: -0.0263372, 0.0245936, -0.0167138, 0.0073999, -0.0337372, 0.0413073
9: -0.0261713, 0.0426285, -0.0171620, 0.0236212, -0.0497925, 0.0597904

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=25, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_B1_B2_B1_B2_A1_A2_A1

### Relational analysis result of IS_B1_B2_B1_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0389630, upper bound: 0.0391525
time: 3.15 seconds

## Relational analysis of IS_B1_B2_B1_B2_A1_A2_A2

### Relational analysis result of IS_B1_B2_B1_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0389630, upper bound: 0.0391525
time: 2.27 seconds

## BFS IS instance: IS_B1_B2_B1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0674993, 0.0575914, -0.0127249, 0.0211417, -0.0886409, 0.0703163
1: -0.0452898, 0.0384938, -0.0111069, 0.0143605, -0.0596503, 0.0496007
2: -0.0981525, 0.0396159, -0.0269850, 0.0166598, -0.1148123, 0.0666009
3: 0.9947226, 1.0378739, 1.0046966, 1.0181364, -0.0234138, 0.0331773
4: -0.0247571, 0.0754903, -0.0042589, 0.0188851, -0.0436422, 0.0797492
5: -0.0371626, 0.1121070, -0.0029642, 0.0492277, -0.0863903, 0.1150711
6: -0.0900687, 0.0667831, -0.0336692, 0.0056171, -0.0956858, 0.1004523
7: -0.0737062, 0.0004235, -0.0338720, -0.0019661, -0.0716885, 0.0342954
8: -0.0436212, 0.0628560, -0.0167138, 0.0073999, -0.0510212, 0.0795698
9: -0.0472143, 0.0649771, -0.0171620, 0.0236212, -0.0708355, 0.0821391

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=25, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of IS_B1_B2_B1_B2_A2_A1_A1

### Relational analysis result of IS_B1_B2_B1_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0384592, upper bound: 0.0381368
time: 2.32 seconds

## Relational analysis of IS_B1_B2_B1_B2_A2_A1_A2

### Relational analysis result of IS_B1_B2_B1_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0382709, upper bound: 0.0380967
time: 2.27 seconds

## BFS IS instance: IS_B1_B2_B1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0866380, 0.0767694, -0.0127249, 0.0211417, -0.1077796, 0.0894944
1: -0.0560291, 0.0457342, -0.0111069, 0.0143605, -0.0703895, 0.0568411
2: -0.1182727, 0.0551352, -0.0269850, 0.0166598, -0.1349325, 0.0821202
3: 0.9892174, 1.0427907, 1.0046966, 1.0181364, -0.0289190, 0.0380940
4: -0.0338728, 0.0968953, -0.0042589, 0.0188851, -0.0527579, 0.1011542
5: -0.0521233, 0.1300275, -0.0029642, 0.0492277, -0.1013510, 0.1329916
6: -0.1075745, 0.0904847, -0.0336692, 0.0056171, -0.1131916, 0.1241539
7: -0.0886654, 0.0047421, -0.0338720, -0.0019661, -0.0866428, 0.0386141
8: -0.0521513, 0.0874271, -0.0167138, 0.0073999, -0.0595512, 0.1041408
9: -0.0607237, 0.0772413, -0.0171620, 0.0236212, -0.0843449, 0.0944033

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=25, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B1_B2_B1_B2_A2_A2_A1

### Relational analysis result of IS_B1_B2_B1_B2_A2_A2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0359014, upper bound: 0.0370907
time: 2.02 seconds

## Relational analysis of IS_B1_B2_B1_B2_A2_A2_A2

### Relational analysis result of IS_B1_B2_B1_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0389076, upper bound: 0.0388730
time: 2.28 seconds

## BFS IS instance: IS_B1_B2_B2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0184003, 0.0247087, -0.0230304, 0.0284927, -0.0468930, 0.0477391
1: -0.0142917, 0.0184406, -0.0162998, 0.0236628, -0.0379545, 0.0347403
2: -0.0379772, 0.0169018, -0.0496224, 0.0169951, -0.0549724, 0.0665242
3: 1.0039140, 1.0212539, 1.0034908, 1.0260203, -0.0221063, 0.0177631
4: -0.0047927, 0.0266384, -0.0043021, 0.0345304, -0.0393231, 0.0309405
5: -0.0057944, 0.0603430, -0.0095762, 0.0705162, -0.0763106, 0.0699192
6: -0.0428485, 0.0087134, -0.0527207, 0.0105316, -0.0533801, 0.0614341
7: -0.0390547, -0.0015068, -0.0442298, -0.0023444, -0.0366615, 0.0426695
8: -0.0201836, 0.0105264, -0.0244290, 0.0127012, -0.0328848, 0.0349554
9: -0.0203145, 0.0312908, -0.0235622, 0.0385131, -0.0588276, 0.0548530

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=24, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 207

## Relational analysis of IS_B1_B2_B2_B1_B1_A1_A1

### Relational analysis result of IS_B1_B2_B2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0386284, upper bound: 0.0380903
time: 2.43 seconds

## Relational analysis of IS_B1_B2_B2_B1_B1_A1_A2

### Relational analysis result of IS_B1_B2_B2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0386284, upper bound: 0.0380903
time: 3.75 seconds

## BFS IS instance: IS_B1_B2_B2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0295456, 0.0289662, -0.0230304, 0.0284927, -0.0580383, 0.0519966
1: -0.0219429, 0.0235464, -0.0162998, 0.0236628, -0.0456057, 0.0398462
2: -0.0542817, 0.0171402, -0.0496224, 0.0169951, -0.0712769, 0.0667626
3: 1.0031314, 1.0260752, 1.0034908, 1.0260203, -0.0228889, 0.0225844
4: -0.0094520, 0.0361665, -0.0043021, 0.0345304, -0.0439823, 0.0404687
5: -0.0112315, 0.0731835, -0.0095762, 0.0705162, -0.0817477, 0.0827597
6: -0.0546507, 0.0224889, -0.0527207, 0.0105316, -0.0651823, 0.0752096
7: -0.0459106, -0.0018946, -0.0442298, -0.0023444, -0.0435093, 0.0422812
8: -0.0250913, 0.0214963, -0.0244290, 0.0127012, -0.0377925, 0.0459253
9: -0.0249409, 0.0401952, -0.0235622, 0.0385131, -0.0634540, 0.0637575

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=24, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 207

## Relational analysis of IS_B1_B2_B2_B1_B1_A2_A1

### Relational analysis result of IS_B1_B2_B2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0386284, upper bound: 0.0382473
time: 2.40 seconds

## Relational analysis of IS_B1_B2_B2_B1_B1_A2_A2

### Relational analysis result of IS_B1_B2_B2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0386284, upper bound: 0.0382473
time: 2.33 seconds

## BFS IS instance: IS_B1_B2_B2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0172096, 0.0242590, -0.0245616, 0.0295819, -0.0467915, 0.0488206
1: -0.0134683, 0.0179285, -0.0170731, 0.0250262, -0.0384945, 0.0350016
2: -0.0362745, 0.0168765, -0.0529589, 0.0170551, -0.0533296, 0.0698354
3: 1.0040003, 1.0207940, 1.0032692, 1.0271891, -0.0231888, 0.0175248
4: -0.0042939, 0.0256284, -0.0043104, 0.0368464, -0.0411402, 0.0299388
5: -0.0052710, 0.0589394, -0.0105591, 0.0736949, -0.0789659, 0.0694985
6: -0.0416206, 0.0072796, -0.0555264, 0.0112504, -0.0528710, 0.0628060
7: -0.0383237, -0.0015108, -0.0457582, -0.0020149, -0.0362636, 0.0441987
8: -0.0197079, 0.0094101, -0.0255862, 0.0135533, -0.0332611, 0.0349963
9: -0.0198297, 0.0303297, -0.0245111, 0.0407524, -0.0605821, 0.0548408

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=24, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 207

## Relational analysis of IS_B1_B2_B2_B1_B2_A1_A1

### Relational analysis result of IS_B1_B2_B2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0386354, upper bound: 0.0380909
time: 3.59 seconds

## Relational analysis of IS_B1_B2_B2_B1_B2_A1_A2

### Relational analysis result of IS_B1_B2_B2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0386354, upper bound: 0.0380909
time: 2.43 seconds

## BFS IS instance: IS_B1_B2_B2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0281881, 0.0284597, -0.0245616, 0.0295819, -0.0577701, 0.0530213
1: -0.0210024, 0.0229599, -0.0170731, 0.0250262, -0.0460287, 0.0400330
2: -0.0523453, 0.0171094, -0.0529589, 0.0170551, -0.0694004, 0.0700683
3: 1.0032274, 1.0255388, 1.0032692, 1.0271891, -0.0239618, 0.0222696
4: -0.0088718, 0.0350348, -0.0043104, 0.0368464, -0.0457182, 0.0393452
5: -0.0106207, 0.0716229, -0.0105591, 0.0736949, -0.0843156, 0.0821820
6: -0.0532677, 0.0208087, -0.0555264, 0.0112504, -0.0645182, 0.0763351
7: -0.0450896, -0.0018976, -0.0457582, -0.0020149, -0.0430224, 0.0438115
8: -0.0245339, 0.0201432, -0.0255862, 0.0135533, -0.0380872, 0.0457294
9: -0.0243935, 0.0391196, -0.0245111, 0.0407524, -0.0651459, 0.0636307

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=24, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 207

## Relational analysis of IS_B1_B2_B2_B1_B2_A2_A1

### Relational analysis result of IS_B1_B2_B2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0386354, upper bound: 0.0382218
time: 2.54 seconds

## Relational analysis of IS_B1_B2_B2_B1_B2_A2_A2

### Relational analysis result of IS_B1_B2_B2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0386354, upper bound: 0.0382221
time: 2.42 seconds

## BFS IS instance: IS_B1_B2_B2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0152612, 0.0228731, -0.0263180, 0.0308031, -0.0460644, 0.0491911
1: -0.0124737, 0.0161797, -0.0179872, 0.0264646, -0.0389383, 0.0341669
2: -0.0320014, 0.0168046, -0.0566467, 0.0171320, -0.0491334, 0.0734514
3: 1.0042562, 1.0192742, 1.0030247, 1.0282885, -0.0240322, 0.0162495
4: -0.0042834, 0.0226494, -0.0043224, 0.0394842, -0.0437676, 0.0269718
5: -0.0039214, 0.0549156, -0.0115015, 0.0774505, -0.0813719, 0.0664170
6: -0.0379680, 0.0063556, -0.0586496, 0.0119422, -0.0499102, 0.0650052
7: -0.0363744, -0.0015506, -0.0475124, -0.0020434, -0.0342864, 0.0459074
8: -0.0181891, 0.0084225, -0.0267864, 0.0144231, -0.0326122, 0.0352089
9: -0.0185983, 0.0275085, -0.0255562, 0.0433528, -0.0619510, 0.0530647

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=25, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 207

## Relational analysis of IS_B1_B2_B2_B2_A1_A1_A1

### Relational analysis result of IS_B1_B2_B2_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0386315, upper bound: 0.0381076
time: 3.27 seconds

## Relational analysis of IS_B1_B2_B2_B2_A1_A1_A2

### Relational analysis result of IS_B1_B2_B2_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0386315, upper bound: 0.0381076
time: 2.51 seconds

## BFS IS instance: IS_B1_B2_B2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0251555, 0.0272347, -0.0263180, 0.0308031, -0.0559587, 0.0535527
1: -0.0189597, 0.0213880, -0.0179872, 0.0264646, -0.0454243, 0.0393752
2: -0.0476551, 0.0170508, -0.0566467, 0.0171320, -0.0647871, 0.0736975
3: 1.0034485, 1.0239513, 1.0030247, 1.0282885, -0.0248400, 0.0209266
4: -0.0076693, 0.0322995, -0.0043224, 0.0394842, -0.0471535, 0.0366219
5: -0.0088660, 0.0681143, -0.0115015, 0.0774505, -0.0863165, 0.0796158
6: -0.0497815, 0.0170561, -0.0586496, 0.0119422, -0.0617237, 0.0757057
7: -0.0431510, -0.0019402, -0.0475124, -0.0020434, -0.0410558, 0.0455175
8: -0.0229829, 0.0171525, -0.0267864, 0.0144231, -0.0374060, 0.0439389
9: -0.0230459, 0.0366551, -0.0255562, 0.0433528, -0.0663987, 0.0622113

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=25, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 207

## Relational analysis of IS_B1_B2_B2_B2_A1_A2_A1

### Relational analysis result of IS_B1_B2_B2_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0386315, upper bound: 0.0382407
time: 2.59 seconds

## Relational analysis of IS_B1_B2_B2_B2_A1_A2_A2

### Relational analysis result of IS_B1_B2_B2_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0386315, upper bound: 0.0382408
time: 2.36 seconds

## BFS IS instance: IS_B1_B2_B2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0171793, 0.0242414, -0.0255657, 0.0302717, -0.0474510, 0.0498071
1: -0.0134465, 0.0179389, -0.0176037, 0.0258050, -0.0392515, 0.0355426
2: -0.0362532, 0.0168737, -0.0550180, 0.0171045, -0.0533578, 0.0718917
3: 1.0039868, 1.0208360, 1.0031197, 1.0277522, -0.0237653, 0.0177163
4: -0.0042933, 0.0255903, -0.0043183, 0.0383467, -0.0426400, 0.0299086
5: -0.0053068, 0.0588372, -0.0110446, 0.0758838, -0.0811906, 0.0698818
6: -0.0415920, 0.0073040, -0.0572755, 0.0116076, -0.0531996, 0.0645795
7: -0.0382968, -0.0011852, -0.0467557, -0.0020465, -0.0362077, 0.0455194
8: -0.0197290, 0.0094412, -0.0262385, 0.0140338, -0.0337627, 0.0356798
9: -0.0198252, 0.0302747, -0.0250955, 0.0422628, -0.0620879, 0.0553701

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=25, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 207

## Relational analysis of IS_B1_B2_B2_B2_A2_A1_A1

### Relational analysis result of IS_B1_B2_B2_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0386382, upper bound: 0.0381128
time: 3.26 seconds

## Relational analysis of IS_B1_B2_B2_B2_A2_A1_A2

### Relational analysis result of IS_B1_B2_B2_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0386382, upper bound: 0.0381128
time: 2.19 seconds

## BFS IS instance: IS_B1_B2_B2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0281500, 0.0284405, -0.0255657, 0.0302717, -0.0584218, 0.0540061
1: -0.0209734, 0.0229709, -0.0176037, 0.0258050, -0.0467784, 0.0405746
2: -0.0523189, 0.0171082, -0.0550180, 0.0171045, -0.0694235, 0.0721262
3: 1.0032114, 1.0255814, 1.0031197, 1.0277522, -0.0245408, 0.0224617
4: -0.0088673, 0.0349915, -0.0043183, 0.0383467, -0.0472139, 0.0393098
5: -0.0106588, 0.0715153, -0.0110446, 0.0758838, -0.0865426, 0.0825599
6: -0.0532355, 0.0208228, -0.0572755, 0.0116076, -0.0648431, 0.0780983
7: -0.0450588, -0.0016257, -0.0467557, -0.0020465, -0.0429649, 0.0450813
8: -0.0245642, 0.0201409, -0.0262385, 0.0140338, -0.0385980, 0.0463794
9: -0.0243849, 0.0390697, -0.0250955, 0.0422628, -0.0666476, 0.0641652

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=25, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 207

## Relational analysis of IS_B1_B2_B2_B2_A2_A2_A1

### Relational analysis result of IS_B1_B2_B2_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0386382, upper bound: 0.0382406
time: 2.98 seconds

## Relational analysis of IS_B1_B2_B2_B2_A2_A2_A2

### Relational analysis result of IS_B1_B2_B2_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0386382, upper bound: 0.0382406
time: 2.36 seconds

## BFS IS instance: IS_B2_A1_B1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0068374, 0.0166737, -0.0303903, 0.0293569, -0.0361942, 0.0470640
1: -0.0078415, 0.0094350, -0.0224821, 0.0241280, -0.0319696, 0.0319171
2: -0.0137121, 0.0164045, -0.0557899, 0.0171452, -0.0308573, 0.0721944
3: 1.0055252, 1.0147557, 1.0030990, 1.0267181, -0.0211929, 0.0116568
4: -0.0042212, 0.0099174, -0.0097361, 0.0370410, -0.0412622, 0.0196535
5: -0.0012505, 0.0356549, -0.0119325, 0.0741645, -0.0754150, 0.0475874
6: -0.0238212, 0.0029969, -0.0558394, 0.0235406, -0.0473617, 0.0588363
7: -0.0277465, -0.0018907, -0.0465110, -0.0020776, -0.0256264, 0.0445727
8: -0.0131211, 0.0051233, -0.0256777, 0.0223102, -0.0354313, 0.0308010
9: -0.0135701, 0.0142430, -0.0253832, 0.0409096, -0.0544797, 0.0396262

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=7, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of IS_B2_A1_B1_B1_A1_A1_B1

### Relational analysis result of IS_B2_A1_B1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0382102, upper bound: 0.0388517
time: 2.22 seconds

## Relational analysis of IS_B2_A1_B1_B1_A1_A1_B2

### Relational analysis result of IS_B2_A1_B1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0382079, upper bound: 0.0387430
time: 2.51 seconds

## BFS IS instance: IS_B2_A1_B1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0256581, 0.0274161, -0.0303903, 0.0293569, -0.0550149, 0.0578064
1: -0.0193110, 0.0216156, -0.0224821, 0.0241280, -0.0434390, 0.0440978
2: -0.0483786, 0.0170655, -0.0557899, 0.0171452, -0.0655238, 0.0728554
3: 1.0033865, 1.0241690, 1.0030990, 1.0267181, -0.0233316, 0.0210700
4: -0.0078965, 0.0327116, -0.0097361, 0.0370410, -0.0449376, 0.0424476
5: -0.0091261, 0.0686419, -0.0119325, 0.0741645, -0.0832906, 0.0805745
6: -0.0503100, 0.0177174, -0.0558394, 0.0235406, -0.0738506, 0.0735568
7: -0.0434550, -0.0016809, -0.0465110, -0.0020776, -0.0413043, 0.0447492
8: -0.0232274, 0.0176687, -0.0256777, 0.0223102, -0.0455376, 0.0433464
9: -0.0232557, 0.0370276, -0.0253832, 0.0409096, -0.0641653, 0.0624109

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_B2_A1_B1_B1_A1_A2_B1

### Relational analysis result of IS_B2_A1_B1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0386039, upper bound: 0.0395944
time: 2.19 seconds

## Relational analysis of IS_B2_A1_B1_B1_A1_A2_B2

### Relational analysis result of IS_B2_A1_B1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0386042, upper bound: 0.0395469
time: 2.20 seconds

## BFS IS instance: IS_B2_A1_B1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0316771, 0.0302032, -0.0303903, 0.0293569, -0.0610340, 0.0605934
1: -0.0231595, 0.0255748, -0.0224821, 0.0241280, -0.0472875, 0.0480569
2: -0.0588682, 0.0171202, -0.0557899, 0.0171452, -0.0760135, 0.0729102
3: 1.0030806, 1.0285366, 1.0030990, 1.0267181, -0.0236375, 0.0254376
4: -0.0098988, 0.0388848, -0.0097361, 0.0370410, -0.0469398, 0.0486209
5: -0.0137545, 0.0759342, -0.0119325, 0.0741645, -0.0879190, 0.0878668
6: -0.0585022, 0.0249521, -0.0558394, 0.0235406, -0.0820427, 0.0807915
7: -0.0476547, -0.0018794, -0.0465110, -0.0020776, -0.0455285, 0.0445829
8: -0.0271340, 0.0234518, -0.0256777, 0.0223102, -0.0494442, 0.0491296
9: -0.0263312, 0.0422488, -0.0253832, 0.0409096, -0.0672408, 0.0676320

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_B2_A1_B1_B1_A2_A1_A1

### Relational analysis result of IS_B2_A1_B1_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0378977, upper bound: 0.0386157
time: 2.36 seconds

## Relational analysis of IS_B2_A1_B1_B1_A2_A1_A2

### Relational analysis result of IS_B2_A1_B1_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0379188, upper bound: 0.0386224
time: 2.28 seconds

## BFS IS instance: IS_B2_A1_B1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0649808, 0.0455211, -0.0303903, 0.0293569, -0.0943377, 0.0759114
1: -0.0482010, 0.0377873, -0.0224821, 0.0241280, -0.0723291, 0.0602695
2: -0.1006792, 0.0314340, -0.0557899, 0.0171452, -0.1178244, 0.0872239
3: 0.9974464, 1.0391507, 1.0030990, 1.0267181, -0.0292718, 0.0360518
4: -0.0213230, 0.0673576, -0.0097361, 0.0370410, -0.0583641, 0.0770936
5: -0.0313302, 0.1161956, -0.0119325, 0.0741645, -0.1054948, 0.1281281
6: -0.0897675, 0.0581490, -0.0558394, 0.0235406, -0.1133080, 0.1139884
7: -0.0695914, 0.0015639, -0.0465110, -0.0020776, -0.0674354, 0.0480750
8: -0.0459148, 0.0503920, -0.0256777, 0.0223102, -0.0682250, 0.0760697
9: -0.0399719, 0.0638060, -0.0253832, 0.0409096, -0.0808814, 0.0891893

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=7, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_B2_A1_B1_B1_A2_A2_B1

### Relational analysis result of IS_B2_A1_B1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0379168, upper bound: 0.0392903
time: 2.25 seconds

## Relational analysis of IS_B2_A1_B1_B1_A2_A2_B2

### Relational analysis result of IS_B2_A1_B1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0379188, upper bound: 0.0392777
time: 2.47 seconds

## BFS IS instance: IS_B2_A1_B1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0314599, 0.0295703, -0.0532492, 0.0382560, -0.0697159, 0.0828196
1: -0.0233259, 0.0241550, -0.0380887, 0.0350142, -0.0583401, 0.0622438
2: -0.0566849, 0.0172011, -0.0897981, 0.0176068, -0.0742917, 0.1069992
3: 1.0029342, 1.0265055, 1.0015384, 1.0371598, -0.0342256, 0.0249671
4: -0.0103893, 0.0375452, -0.0191186, 0.0569325, -0.0673219, 0.0566638
5: -0.0118027, 0.0752334, -0.0237024, 0.1005915, -0.1123942, 0.0989358
6: -0.0562535, 0.0249629, -0.0806857, 0.0517926, -0.1080461, 0.1056486
7: -0.0469572, -0.0014962, -0.0607435, -0.0021853, -0.0447366, 0.0591548
8: -0.0256824, 0.0234776, -0.0362494, 0.0452245, -0.0709068, 0.0597270
9: -0.0256043, 0.0415993, -0.0351016, 0.0592898, -0.0848941, 0.0767009

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=24, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 207

## Relational analysis of IS_B2_A1_B1_B2_B2_B1_A1

### Relational analysis result of IS_B2_A1_B1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0391260, upper bound: 0.0390701
time: 2.13 seconds

## Relational analysis of IS_B2_A1_B1_B2_B2_B1_A2

### Relational analysis result of IS_B2_A1_B1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0391260, upper bound: 0.0390701
time: 2.20 seconds

## BFS IS instance: IS_B2_A1_B1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0302972, 0.0291283, -0.0562004, 0.0394143, -0.0697115, 0.0853287
1: -0.0225258, 0.0236221, -0.0400950, 0.0364421, -0.0589679, 0.0637170
2: -0.0549823, 0.0171761, -0.0942134, 0.0176658, -0.0726481, 0.1113895
3: 1.0030179, 1.0260043, 1.0013112, 1.0385696, -0.0355517, 0.0246931
4: -0.0099006, 0.0365533, -0.0203196, 0.0595051, -0.0694057, 0.0568729
5: -0.0112315, 0.0739044, -0.0252591, 0.1040053, -0.1152367, 0.0991635
6: -0.0550215, 0.0235151, -0.0838974, 0.0554467, -0.1104682, 0.1074125
7: -0.0462402, -0.0015002, -0.0625746, -0.0018827, -0.0443309, 0.0609848
8: -0.0251676, 0.0223135, -0.0376432, 0.0481852, -0.0733528, 0.0599567
9: -0.0251212, 0.0406767, -0.0363577, 0.0616775, -0.0867987, 0.0770343

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=24, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 207

## Relational analysis of IS_B2_A1_B1_B2_B2_B2_A1

### Relational analysis result of IS_B2_A1_B1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0391260, upper bound: 0.0390486
time: 2.54 seconds

## Relational analysis of IS_B2_A1_B1_B2_B2_B2_A2

### Relational analysis result of IS_B2_A1_B1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0391260, upper bound: 0.0390486
time: 2.23 seconds

## BFS IS instance: IS_B2_A1_B2_A2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0233993, 0.0264647, -0.0334009, 0.0303626, -0.0537618, 0.0598657
1: -0.0178137, 0.0203358, -0.0246391, 0.0252590, -0.0430727, 0.0449749
2: -0.0447389, 0.0170292, -0.0597974, 0.0172201, -0.0619590, 0.0768266
3: 1.0035257, 1.0228052, 1.0027655, 1.0276970, -0.0241712, 0.0200397
4: -0.0070361, 0.0305973, -0.0110242, 0.0394347, -0.0464708, 0.0416215
5: -0.0076550, 0.0660263, -0.0130920, 0.0775758, -0.0852308, 0.0791184
6: -0.0475650, 0.0149128, -0.0586478, 0.0271565, -0.0747215, 0.0735606
7: -0.0419766, -0.0017382, -0.0481832, -0.0014777, -0.0404501, 0.0463623
8: -0.0219578, 0.0154236, -0.0268985, 0.0252308, -0.0471886, 0.0423221
9: -0.0222018, 0.0351682, -0.0265496, 0.0431157, -0.0653175, 0.0617178

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 207

## Relational analysis of IS_B2_A1_B2_A2_A1_A1_B1

### Relational analysis result of IS_B2_A1_B2_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0391354, upper bound: 0.0393532
time: 2.46 seconds

## Relational analysis of IS_B2_A1_B2_A2_A1_A1_B2

### Relational analysis result of IS_B2_A1_B2_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0391354, upper bound: 0.0393532
time: 2.65 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 6.47 seconds
IS_B1_B1_B1_A1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 6.47
Output dim: 3, lower bound: -0.0356515, upper bound: 0.0360434
IS_B1_B1_B1_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.47
Output dim: 3, lower bound: -0.0385089, upper bound: 0.0380754
IS_B1_B1_B1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.47
Output dim: 3, lower bound: -0.0385614, upper bound: 0.0381410
IS_B1_B1_B1_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.47
Output dim: 3, lower bound: -0.0385614, upper bound: 0.0381410
IS_B1_B1_B1_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 6.47
Output dim: 3, lower bound: -0.0383680, upper bound: 0.0380875
IS_B1_B1_B1_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 6.47
Output dim: 3, lower bound: -0.0383680, upper bound: 0.0380875
IS_B1_B1_B1_A1_A2_A2_A1, status: Status.VERIFIED, split count: 7, time: 6.47
Output dim: 3, lower bound: -0.0353159, upper bound: 0.0358989
IS_B1_B1_B1_A1_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 6.47
Output dim: 3, lower bound: -0.0383112, upper bound: 0.0380213
IS_B1_B1_B1_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.47
Output dim: 3, lower bound: -0.0385585, upper bound: 0.0381408
IS_B1_B1_B1_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.47
Output dim: 3, lower bound: -0.0385585, upper bound: 0.0381408
IS_B1_B1_B1_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.47
Output dim: 3, lower bound: -0.0385585, upper bound: 0.0381408
IS_B1_B1_B1_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.47
Output dim: 3, lower bound: -0.0385585, upper bound: 0.0381408
IS_B1_B1_B1_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 6.47
Output dim: 3, lower bound: -0.0383630, upper bound: 0.0380873
IS_B1_B1_B1_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 6.47
Output dim: 3, lower bound: -0.0383630, upper bound: 0.0380873
IS_B1_B1_B1_A2_A2_A2_A1, status: Status.VERIFIED, split count: 7, time: 6.47
Output dim: 3, lower bound: -0.0352890, upper bound: 0.0358887
IS_B1_B1_B1_A2_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 6.47
Output dim: 3, lower bound: -0.0383059, upper bound: 0.0380213
IS_B1_B1_B2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 6.47
Output dim: 3, lower bound: -0.0383847, upper bound: 0.0381755
IS_B1_B1_B2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 6.47
Output dim: 3, lower bound: -0.0383847, upper bound: 0.0381755
IS_B1_B1_B2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 6.47
Output dim: 3, lower bound: -0.0380217, upper bound: 0.0378374
IS_B1_B1_B2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 6.47
Output dim: 3, lower bound: -0.0380480, upper bound: 0.0378374
IS_B1_B1_B2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 6.47
Output dim: 3, lower bound: -0.0383900, upper bound: 0.0381925
IS_B1_B1_B2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 6.47
Output dim: 3, lower bound: -0.0383900, upper bound: 0.0381925
IS_B1_B1_B2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 6.47
Output dim: 3, lower bound: -0.0380270, upper bound: 0.0378584
IS_B1_B1_B2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 6.47
Output dim: 3, lower bound: -0.0380546, upper bound: 0.0378584
IS_B1_B2_B1_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 6.47
Output dim: 3, lower bound: -0.0389626, upper bound: 0.0391095
IS_B1_B2_B1_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 6.47
Output dim: 3, lower bound: -0.0389626, upper bound: 0.0391095
IS_B1_B2_B1_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 6.47
Output dim: 3, lower bound: -0.0389626, upper bound: 0.0391864
IS_B1_B2_B1_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 6.47
Output dim: 3, lower bound: -0.0389626, upper bound: 0.0391864
IS_B1_B2_B1_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 6.47
Output dim: 3, lower bound: -0.0384533, upper bound: 0.0381372
IS_B1_B2_B1_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 6.47
Output dim: 3, lower bound: -0.0382680, upper bound: 0.0380991
IS_B1_B2_B1_B1_A2_A2_A1, status: Status.VERIFIED, split count: 7, time: 6.47
Output dim: 3, lower bound: -0.0359013, upper bound: 0.0370929
IS_B1_B2_B1_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 6.47
Output dim: 3, lower bound: -0.0389022, upper bound: 0.0388818
IS_B1_B2_B1_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 6.47
Output dim: 3, lower bound: -0.0389630, upper bound: 0.0390924
IS_B1_B2_B1_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 6.47
Output dim: 3, lower bound: -0.0389630, upper bound: 0.0390924
IS_B1_B2_B1_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 6.47
Output dim: 3, lower bound: -0.0389630, upper bound: 0.0391525
IS_B1_B2_B1_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 6.47
Output dim: 3, lower bound: -0.0389630, upper bound: 0.0391525
IS_B1_B2_B1_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 6.47
Output dim: 3, lower bound: -0.0384592, upper bound: 0.0381368
IS_B1_B2_B1_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 6.47
Output dim: 3, lower bound: -0.0382709, upper bound: 0.0380967
IS_B1_B2_B1_B2_A2_A2_A1, status: Status.VERIFIED, split count: 7, time: 6.47
Output dim: 3, lower bound: -0.0359014, upper bound: 0.0370907
IS_B1_B2_B1_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 6.47
Output dim: 3, lower bound: -0.0389076, upper bound: 0.0388730
IS_B1_B2_B2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 6.47
Output dim: 3, lower bound: -0.0386284, upper bound: 0.0380903
IS_B1_B2_B2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 6.47
Output dim: 3, lower bound: -0.0386284, upper bound: 0.0380903
IS_B1_B2_B2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 6.47
Output dim: 3, lower bound: -0.0386284, upper bound: 0.0382473
IS_B1_B2_B2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 6.47
Output dim: 3, lower bound: -0.0386284, upper bound: 0.0382473
IS_B1_B2_B2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 6.47
Output dim: 3, lower bound: -0.0386354, upper bound: 0.0380909
IS_B1_B2_B2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 6.47
Output dim: 3, lower bound: -0.0386354, upper bound: 0.0380909
IS_B1_B2_B2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 6.47
Output dim: 3, lower bound: -0.0386354, upper bound: 0.0382218
IS_B1_B2_B2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 6.47
Output dim: 3, lower bound: -0.0386354, upper bound: 0.0382221
IS_B1_B2_B2_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 6.47
Output dim: 3, lower bound: -0.0386315, upper bound: 0.0381076
IS_B1_B2_B2_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 6.47
Output dim: 3, lower bound: -0.0386315, upper bound: 0.0381076
IS_B1_B2_B2_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 6.47
Output dim: 3, lower bound: -0.0386315, upper bound: 0.0382407
IS_B1_B2_B2_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 6.47
Output dim: 3, lower bound: -0.0386315, upper bound: 0.0382408
IS_B1_B2_B2_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 6.47
Output dim: 3, lower bound: -0.0386382, upper bound: 0.0381128
IS_B1_B2_B2_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 6.47
Output dim: 3, lower bound: -0.0386382, upper bound: 0.0381128
IS_B1_B2_B2_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 6.47
Output dim: 3, lower bound: -0.0386382, upper bound: 0.0382406
IS_B1_B2_B2_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 6.47
Output dim: 3, lower bound: -0.0386382, upper bound: 0.0382406
IS_B2_A1_B1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 6.47
Output dim: 3, lower bound: -0.0382102, upper bound: 0.0388517
IS_B2_A1_B1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 6.47
Output dim: 3, lower bound: -0.0382079, upper bound: 0.0387430
IS_B2_A1_B1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 6.47
Output dim: 3, lower bound: -0.0386039, upper bound: 0.0395944
IS_B2_A1_B1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 6.47
Output dim: 3, lower bound: -0.0386042, upper bound: 0.0395469
IS_B2_A1_B1_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 6.47
Output dim: 3, lower bound: -0.0378977, upper bound: 0.0386157
IS_B2_A1_B1_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 6.47
Output dim: 3, lower bound: -0.0379188, upper bound: 0.0386224
IS_B2_A1_B1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 6.47
Output dim: 3, lower bound: -0.0379168, upper bound: 0.0392903
IS_B2_A1_B1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 6.47
Output dim: 3, lower bound: -0.0379188, upper bound: 0.0392777
IS_B2_A1_B1_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.47
Output dim: 3, lower bound: -0.0391260, upper bound: 0.0390701
IS_B2_A1_B1_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.47
Output dim: 3, lower bound: -0.0391260, upper bound: 0.0390701
IS_B2_A1_B1_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.47
Output dim: 3, lower bound: -0.0391260, upper bound: 0.0390486
IS_B2_A1_B1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.47
Output dim: 3, lower bound: -0.0391260, upper bound: 0.0390486
IS_B2_A1_B2_A2_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 6.47
Output dim: 3, lower bound: -0.0391354, upper bound: 0.0393532
IS_B2_A1_B2_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 6.47
Output dim: 3, lower bound: -0.0391354, upper bound: 0.0393532
IS_B2_A1_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 6.47
Output dim: 3, lower bound: -0.0391354, upper bound: 0.0393532
IS_B2_A1_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 6.47
Output dim: 3, lower bound: -0.0394584, upper bound: 0.0393807
IS_B2_A1_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 6.47
Output dim: 3, lower bound: -0.0391442, upper bound: 0.0393559
IS_B2_A2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 6.47
Output dim: 3, lower bound: -0.0392687, upper bound: 0.0391755
IS_B2_A2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 6.47
Output dim: 3, lower bound: -0.0390093, upper bound: 0.0391710
IS_B2_A2_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 6.47
Output dim: 3, lower bound: -0.0391416, upper bound: 0.0395897
IS_B2_A2_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 6.47
Output dim: 3, lower bound: -0.0391413, upper bound: 0.0394414
IS_B2_A2_A2_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.47
Output dim: 3, lower bound: -0.0393301, upper bound: 0.0393989
IS_B2_A2_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.47
Output dim: 3, lower bound: -0.0393301, upper bound: 0.0393989
IS_B2_A2_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.47
Output dim: 3, lower bound: -0.0397541, upper bound: 0.0395347
IS_B2_A2_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.47
Output dim: 3, lower bound: -0.0397541, upper bound: 0.0395347
IS_B2_A2_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.47
Output dim: 3, lower bound: -0.0383102, upper bound: 0.0390702
IS_B2_A2_A2_A2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 6.47
Output dim: 3, lower bound: -0.0378977, upper bound: 0.0393483
IS_B2_A2_A2_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 6.47
Output dim: 3, lower bound: -0.0381837, upper bound: 0.0395085

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 4.37 + 598.03 = 602.40 seconds
