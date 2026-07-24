## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 2.411746211


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716)
1: (-0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653)
2: (-0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282)
3: (-1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487)
4: (-1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829)
5: (-1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234)
6: (-1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365)
7: (-1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017)
8: (-1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066)
9: (-1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.43 + 3.82 = 5.25 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -2.4863377, upper bound: 2.4863377

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -2.2817157, upper bound: 2.3617292
time: 1.50 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4861420, upper bound: 2.4861420
time: 2.06 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 3.71 seconds
IS_A1, status: Status.VERIFIED, split count: 1, time: 3.71
Output dim: 8, lower bound: -2.2817157, upper bound: 2.3617292
IS_A2, status: Status.UNKNOWN, split count: 1, time: 3.71
Output dim: 8, lower bound: -2.4861420, upper bound: 2.4861420

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -1.0353376, 0.9610157, -1.1449089, 1.0237626, -2.0591002, 2.1059246
1: -0.8703837, 0.8642530, -0.9533735, 0.9367918, -1.8071755, 1.8176265
2: -0.8845018, 1.0364182, -0.9885503, 1.1024778, -1.9869795, 2.0249686
3: -1.0000892, 0.9567776, -1.1323893, 1.0104594, -2.0105486, 2.0891669
4: -1.1045318, 0.8888527, -1.2352548, 0.9672282, -2.0717599, 2.1241074
5: -0.9205687, 0.9245936, -1.0193650, 0.9940586, -1.9146273, 1.9439585
6: -0.9308109, 0.9743273, -1.0222397, 1.0629967, -1.9938077, 1.9965670
7: -0.9863638, 1.0524640, -1.0907527, 1.1420490, -2.1284127, 2.1432166
8: -1.1343669, 1.3051724, -1.3188362, 1.3299705, -2.4643373, 2.6240087
9: -0.9701360, 1.0525401, -1.0492049, 1.1355765, -2.1057124, 2.1017451

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=71, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=127, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4818443, upper bound: 2.4716494
time: 1.70 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4713266, upper bound: 2.4713266
time: 1.66 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 4.79 seconds
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 4.79
Output dim: 8, lower bound: -2.4818443, upper bound: 2.4716494
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 4.79
Output dim: 8, lower bound: -2.4713266, upper bound: 2.4713266

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: -0.5181744, 0.6507804, -1.0908915, 0.9910498, -1.5092242, 1.7416719
1: -0.4727534, 0.4883611, -0.9098534, 0.8986187, -1.3713721, 1.3982146
2: -0.4061381, 0.6617128, -0.9363099, 1.0676720, -1.4738102, 1.5980227
3: -0.3961257, 0.5583881, -1.0679227, 0.9718989, -1.3680246, 1.6263108
4: -0.5116181, 0.4863809, -1.1711588, 0.9269629, -1.4385810, 1.6575396
5: -0.4724057, 0.5609036, -0.9690905, 0.9593685, -1.4317741, 1.5299940
6: -0.4572800, 0.5668856, -0.9749643, 1.0179003, -1.4751804, 1.5418499
7: -0.5134760, 0.5969673, -1.0394449, 1.0975792, -1.6110553, 1.6364123
8: -0.1334802, 1.1498952, -1.2196703, 1.3106424, -1.4441226, 2.3695655
9: -0.5448413, 0.6321118, -1.0045564, 1.0920416, -1.6368830, 1.6366682

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=71, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=131, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_A1_B1

### Relational analysis result of IS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4354129, upper bound: 2.4001033
time: 1.74 seconds

## Relational analysis of IS_A2_A1_B2

### Relational analysis result of IS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4469294, upper bound: 2.4345021
time: 1.60 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: -1.1132535, 1.0560507, -0.8451622, 0.8477327, -1.9609861, 1.9012129
1: -0.9138149, 0.8970931, -0.7212036, 0.7284132, -1.6422281, 1.6182966
2: -0.9623706, 1.0841529, -0.7009112, 0.9098111, -1.8721817, 1.7850642
3: -1.1094935, 0.8769898, -0.7803892, 0.8110399, -1.9205334, 1.6573792
4: -1.2106135, 0.9380180, -0.8811504, 0.7458646, -1.9564781, 1.8191683
5: -0.9837235, 0.9812996, -0.7433056, 0.7988237, -1.7825472, 1.7246051
6: -0.9938590, 1.0428290, -0.7599764, 0.8188689, -1.8127279, 1.8028054
7: -1.0682783, 1.1253543, -0.8049903, 0.8975774, -1.9658557, 1.9303446
8: -1.2722735, 1.3103857, -0.7719377, 1.2375937, -2.5098672, 2.0823236
9: -1.0067015, 1.1201277, -0.8105056, 0.8944100, -1.9011114, 1.9306333

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=73, inp2_unstable=71, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=108, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_A2_B1

### Relational analysis result of IS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4713266, upper bound: 2.4713266
time: 1.64 seconds

## Relational analysis of IS_A2_A2_B2

### Relational analysis result of IS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4713266, upper bound: 2.4713266
time: 1.58 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.66 seconds
IS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 4.66
Output dim: 8, lower bound: -2.4354129, upper bound: 2.4001033
IS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 4.66
Output dim: 8, lower bound: -2.4469294, upper bound: 2.4345021
IS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 4.66
Output dim: 8, lower bound: -2.4713266, upper bound: 2.4713266
IS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 4.66
Output dim: 8, lower bound: -2.4713266, upper bound: 2.4713266

## BFS IS instance: IS_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.3173478, 0.4665371, -0.1436929, 0.2568283, -0.5741761, 0.6102300
1: -0.2967142, 0.2997675, -0.1440818, 0.1513046, -0.4480189, 0.4438493
2: -0.2972906, 0.4114707, -0.1559809, 0.2088203, -0.5061109, 0.5674516
3: -0.2193102, 0.3827772, -0.1044473, 0.1857741, -0.4050843, 0.4872244
4: -0.3038100, 0.2969496, -0.1550779, 0.1348786, -0.4386885, 0.4520276
5: -0.3022537, 0.3790665, -0.1386610, 0.1931287, -0.4953824, 0.5177275
6: -0.2840036, 0.3796705, -0.1404694, 0.1835814, -0.4675851, 0.5201399
7: -0.3273391, 0.3669847, -0.1826921, 0.1681871, -0.4955262, 0.5496768
8: 0.2723187, 1.1207510, 0.5866538, 1.0763659, -0.8040472, 0.5340972
9: -0.3668229, 0.4509652, -0.1835557, 0.2659805, -0.6328033, 0.6345209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=33, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=66, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_A1_B1_B1

### Relational analysis result of IS_A2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -2.3935569, upper bound: 2.3407892
time: 1.83 seconds

## Relational analysis of IS_A2_A1_B1_B2

### Relational analysis result of IS_A2_A1_B1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -2.4088916, upper bound: 2.3756514
time: 1.86 seconds

## BFS IS instance: IS_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.4010081, 0.5483533, -0.3651343, 0.5105548, -0.9115629, 0.9134876
1: -0.3700185, 0.3825445, -0.3379239, 0.3458358, -0.7158542, 0.7204683
2: -0.3456911, 0.5118557, -0.3298604, 0.4649336, -0.8106247, 0.8417162
3: -0.2772306, 0.4656217, -0.2468444, 0.4498911, -0.7271216, 0.7124661
4: -0.3793129, 0.3768206, -0.3444784, 0.3454679, -0.7247809, 0.7212991
5: -0.3792559, 0.4500957, -0.3411778, 0.4219283, -0.8011842, 0.7912735
6: -0.3544722, 0.4660694, -0.3274321, 0.4286562, -0.7831284, 0.7935015
7: -0.3983823, 0.4588357, -0.3681895, 0.4190326, -0.8174149, 0.8270253
8: 0.1147439, 1.1325300, 0.1739811, 1.1383959, -1.0236520, 0.9585488
9: -0.4439164, 0.5306185, -0.4216031, 0.4941778, -0.9380941, 0.9522215

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=33, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=41, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_A1_B2_A1

### Relational analysis result of IS_A2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -2.3962127, upper bound: 2.4021425
time: 1.61 seconds

## Relational analysis of IS_A2_A1_B2_A2

### Relational analysis result of IS_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4189251, upper bound: 2.4062067
time: 1.61 seconds

## BFS IS instance: IS_A2_A2_B1

### Backsubstitution after applying IS history:
0: -1.1132535, 1.0560507, -0.5747843, 0.6893958, -1.8026493, 1.6308349
1: -0.9138149, 0.8970931, -0.5155984, 0.5318701, -1.4456849, 1.4126914
2: -0.9623706, 1.0841529, -0.4466929, 0.7143084, -1.6766789, 1.5308459
3: -1.1094935, 0.8769898, -0.4612007, 0.5952795, -1.7047729, 1.3381906
4: -1.2106135, 0.9380180, -0.5732180, 0.5367400, -1.7473536, 1.5112361
5: -0.9837235, 0.9812996, -0.5142132, 0.6119117, -1.5956352, 1.4955127
6: -0.9938590, 1.0428290, -0.5075482, 0.6086887, -1.6025476, 1.5503771
7: -1.0682783, 1.1253543, -0.5654731, 0.6563929, -1.7246712, 1.6908274
8: -1.2722735, 1.3103857, -0.2437143, 1.1576909, -2.4299645, 1.5541000
9: -1.0067015, 1.1201277, -0.5893691, 0.6735519, -1.6802533, 1.7094967

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=33, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=73, inp2_unstable=71, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=71, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 8

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_A2_B1_B1

### Relational analysis result of IS_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4532700, upper bound: 2.4519942
time: 1.53 seconds

## Relational analysis of IS_A2_A2_B1_B2

### Relational analysis result of IS_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4535350, upper bound: 2.4535350
time: 1.48 seconds

## BFS IS instance: IS_A2_A2_B2

### Backsubstitution after applying IS history:
0: -1.1132535, 1.0560507, -1.1929209, 1.0994191, -2.2126727, 2.2489715
1: -0.9138149, 0.8970931, -0.9646458, 0.9531134, -1.8669283, 1.8617389
2: -0.9623706, 1.0841529, -1.0404637, 1.1412833, -2.1036539, 2.1246166
3: -1.1094935, 0.8769898, -1.2031910, 0.9228392, -2.0323327, 2.0801809
4: -1.2106135, 0.9380180, -1.2952347, 0.9989744, -2.2095878, 2.2332528
5: -0.9837235, 0.9812996, -1.0607319, 1.0241019, -2.0078254, 2.0420315
6: -0.9938590, 1.0428290, -1.0686316, 1.1026409, -2.0964999, 2.1114607
7: -1.0682783, 1.1253543, -1.1308079, 1.1962013, -2.2644796, 2.2561622
8: -1.2722735, 1.3103857, -1.4121656, 1.3229702, -2.5952437, 2.7225513
9: -1.0067015, 1.1201277, -1.0664822, 1.1868098, -2.1935112, 2.1866097

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=33, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=73, inp2_unstable=73, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=140, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_A2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4271575, upper bound: 2.3998600
time: 1.64 seconds

## Relational analysis of IS_A2_A2_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4342495, upper bound: 2.4342495
time: 1.62 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.70 seconds
IS_A2_A1_B1_B1, status: Status.VERIFIED, split count: 4, time: 4.70
Output dim: 8, lower bound: -2.3935569, upper bound: 2.3407892
IS_A2_A1_B1_B2, status: Status.VERIFIED, split count: 4, time: 4.70
Output dim: 8, lower bound: -2.4088916, upper bound: 2.3756514
IS_A2_A1_B2_A1, status: Status.VERIFIED, split count: 4, time: 4.70
Output dim: 8, lower bound: -2.3962127, upper bound: 2.4021425
IS_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 4.70
Output dim: 8, lower bound: -2.4189251, upper bound: 2.4062067
IS_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 4.70
Output dim: 8, lower bound: -2.4532700, upper bound: 2.4519942
IS_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 4.70
Output dim: 8, lower bound: -2.4535350, upper bound: 2.4535350
IS_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 4.70
Output dim: 8, lower bound: -2.4271575, upper bound: 2.3998600
IS_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 4.70
Output dim: 8, lower bound: -2.4342495, upper bound: 2.4342495

## BFS IS instance: IS_A2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.2101627, 0.3532345, -0.3158552, 0.4615001, -0.6716628, 0.6690897
1: -0.2055681, 0.2194042, -0.2962081, 0.2996143, -0.5051824, 0.5156123
2: -0.2188228, 0.2935197, -0.2964966, 0.4102832, -0.6291060, 0.5900162
3: -0.1539435, 0.2663252, -0.2168044, 0.3866804, -0.5406238, 0.4831296
4: -0.2205953, 0.1916153, -0.3039118, 0.2937476, -0.5143430, 0.4955271
5: -0.2047403, 0.2710029, -0.2981818, 0.3769743, -0.5817146, 0.5691847
6: -0.1952375, 0.2646858, -0.2827094, 0.3773941, -0.5726316, 0.5473951
7: -0.2453060, 0.2393087, -0.3253846, 0.3643581, -0.6096641, 0.5646933
8: 0.4643646, 1.0926931, 0.2736759, 1.1236317, -0.6592671, 0.8190172
9: -0.2597876, 0.3492618, -0.3673084, 0.4479354, -0.7077230, 0.7165701

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=33, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=6, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=37, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_A1_B2_A2_B1

### Relational analysis result of IS_A2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -2.3945828, upper bound: 2.3726132
time: 1.55 seconds

## Relational analysis of IS_A2_A1_B2_A2_B2

### Relational analysis result of IS_A2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -2.3953878, upper bound: 2.3805562
time: 1.55 seconds

## BFS IS instance: IS_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.9758284, 0.9791362, -0.3074940, 0.4545946, -1.4304230, 1.2866303
1: -0.8085421, 0.8071840, -0.2893153, 0.2922134, -1.1007555, 1.0964993
2: -0.8310717, 1.0005720, -0.2905205, 0.3996400, -1.2307117, 1.2910924
3: -0.9480097, 0.8073722, -0.2136034, 0.3727720, -1.3207817, 1.0209756
4: -1.0468847, 0.8390745, -0.2949924, 0.2873073, -1.3341919, 1.1340668
5: -0.8605543, 0.8936032, -0.2947748, 0.3702957, -1.2308500, 1.1883780
6: -0.8788103, 0.9329001, -0.2739038, 0.3694673, -1.2482777, 1.2068040
7: -0.9376106, 1.0126183, -0.3205884, 0.3553462, -1.2929568, 1.3332067
8: -1.0376896, 1.2799405, 0.2938109, 1.1135156, -2.1512051, 0.9861296
9: -0.9076507, 1.0148265, -0.3549606, 0.4415006, -1.3491514, 1.3697872

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=69, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=124, inp2_unstable=36, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_A2_B1_B1_B1

### Relational analysis result of IS_A2_A2_B1_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -2.4106566, upper bound: 2.3980297
time: 2.04 seconds

## Relational analysis of IS_A2_A2_B1_B1_B2

### Relational analysis result of IS_A2_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4165037, upper bound: 2.4268438
time: 1.61 seconds

## BFS IS instance: IS_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.8328020, 0.8993166, -0.5154521, 0.6774722, -1.5102742, 1.4147687
1: -0.7032540, 0.7132429, -0.4729923, 0.4902826, -1.1935366, 1.1862352
2: -0.6956232, 0.9123711, -0.4112594, 0.6702324, -1.3658556, 1.3236305
3: -0.7780256, 0.7369632, -0.3949560, 0.5613123, -1.3393378, 1.1319191
4: -0.8756414, 0.7354404, -0.5114701, 0.4862465, -1.3618879, 1.2469106
5: -0.7297786, 0.8009800, -0.4721982, 0.5655399, -1.2953186, 1.2731782
6: -0.7578247, 0.8178355, -0.4628181, 0.5722766, -1.3301013, 1.2806536
7: -0.8006594, 0.8980513, -0.5146209, 0.6015550, -1.4022144, 1.4126723
8: -0.7935832, 1.2529750, -0.1745371, 1.1722178, -1.9658010, 1.4275120
9: -0.8049932, 0.9033895, -0.5595489, 0.6452586, -1.4502518, 1.4629383

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=71, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=109, inp2_unstable=56, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_A2_B1_B2_B1

### Relational analysis result of IS_A2_A2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -2.4105255, upper bound: 2.3971950
time: 1.92 seconds

## Relational analysis of IS_A2_A2_B1_B2_B2

### Relational analysis result of IS_A2_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4166631, upper bound: 2.4289478
time: 1.87 seconds

## BFS IS instance: IS_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.5442522, 0.7448650, -0.1702941, 0.3697815, -0.9140337, 0.9151591
1: -0.4991090, 0.5207657, -0.1750406, 0.1922890, -0.6913980, 0.6958064
2: -0.4437059, 0.7169357, -0.2035190, 0.2662780, -0.7099838, 0.9204547
3: -0.4307705, 0.5895276, -0.1306669, 0.2273058, -0.6580763, 0.7201946
4: -0.5526362, 0.5164620, -0.1893187, 0.1620091, -0.7146453, 0.7057807
5: -0.4979086, 0.6042851, -0.1746739, 0.2399264, -0.7378350, 0.7789590
6: -0.5102216, 0.6040712, -0.1914201, 0.2332408, -0.7434624, 0.7954913
7: -0.5473555, 0.6463960, -0.2161589, 0.2232851, -0.7706406, 0.8625549
8: -0.3031471, 1.2180145, 0.4308179, 1.1437316, -1.4468787, 0.7871966
9: -0.6138389, 0.6904352, -0.2691313, 0.3358222, -0.9496611, 0.9595665

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=73, inp2_unstable=68, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=64, inp2_unstable=29, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_A2_B2_B1_A1

### Relational analysis result of IS_A2_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -2.3734284, upper bound: 2.3663331
time: 1.66 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2

### Relational analysis result of IS_A2_A2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -2.4001541, upper bound: 2.3754072
time: 1.81 seconds

## BFS IS instance: IS_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.8063089, 0.8849697, -0.4100027, 0.6280302, -1.4343390, 1.2949724
1: -0.6843646, 0.6967893, -0.3830291, 0.4035364, -1.0879010, 1.0798185
2: -0.6712427, 0.8957824, -0.3674081, 0.5523592, -1.2236018, 1.2631905
3: -0.7466015, 0.7239174, -0.2873243, 0.4847117, -1.2313132, 1.0112417
4: -0.8448620, 0.7160163, -0.3973778, 0.3881576, -1.2330197, 1.1133941
5: -0.7058659, 0.7842637, -0.3928727, 0.4740278, -1.1798937, 1.1771364
6: -0.7361779, 0.7975273, -0.3891228, 0.4915229, -1.2277007, 1.1866500
7: -0.7761143, 0.8762760, -0.4116562, 0.4905226, -1.2666368, 1.2879322
8: -0.7483493, 1.2503929, -0.0105936, 1.1987159, -1.9470652, 1.2609864
9: -0.7864900, 0.8833165, -0.4984327, 0.5755572, -1.3620472, 1.3817492

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=73, inp2_unstable=73, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=104, inp2_unstable=45, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_A2_B2_B2_A1

### Relational analysis result of IS_A2_A2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -2.3812116, upper bound: 2.4011164
time: 1.48 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2

### Relational analysis result of IS_A2_A2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -2.4059603, upper bound: 2.4059603
time: 1.42 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.34 seconds
IS_A2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 5, time: 4.34
Output dim: 8, lower bound: -2.3945828, upper bound: 2.3726132
IS_A2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 5, time: 4.34
Output dim: 8, lower bound: -2.3953878, upper bound: 2.3805562
IS_A2_A2_B1_B1_B1, status: Status.VERIFIED, split count: 5, time: 4.34
Output dim: 8, lower bound: -2.4106566, upper bound: 2.3980297
IS_A2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.34
Output dim: 8, lower bound: -2.4165037, upper bound: 2.4268438
IS_A2_A2_B1_B2_B1, status: Status.VERIFIED, split count: 5, time: 4.34
Output dim: 8, lower bound: -2.4105255, upper bound: 2.3971950
IS_A2_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.34
Output dim: 8, lower bound: -2.4166631, upper bound: 2.4289478
IS_A2_A2_B2_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.34
Output dim: 8, lower bound: -2.3734284, upper bound: 2.3663331
IS_A2_A2_B2_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.34
Output dim: 8, lower bound: -2.4001541, upper bound: 2.3754072
IS_A2_A2_B2_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.34
Output dim: 8, lower bound: -2.3812116, upper bound: 2.4011164
IS_A2_A2_B2_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.34
Output dim: 8, lower bound: -2.4059603, upper bound: 2.4059603

## BFS IS instance: IS_A2_A2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -0.6872444, 0.8220279, -0.1398435, 0.2626013, -0.9498457, 0.9618714
1: -0.6013268, 0.6196252, -0.1439500, 0.1505850, -0.7519118, 0.7635752
2: -0.5608240, 0.8206819, -0.1569118, 0.2085865, -0.7694106, 0.9775938
3: -0.6012737, 0.6664212, -0.1036550, 0.1851355, -0.7864091, 0.7700762
4: -0.7098660, 0.6269656, -0.1510804, 0.1324330, -0.8422990, 0.7780460
5: -0.6035425, 0.7082616, -0.1366209, 0.1937581, -0.7973006, 0.8448825
6: -0.6298481, 0.7083503, -0.1401921, 0.1836783, -0.8135263, 0.8485424
7: -0.6699082, 0.7759130, -0.1822176, 0.1685469, -0.8384551, 0.9581306
8: -0.5481834, 1.2318909, 0.5795054, 1.0805454, -1.6287289, 0.6523855
9: -0.7044834, 0.7945080, -0.1867295, 0.2688506, -0.9733340, 0.9812375

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=31, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=69, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=96, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_A2_B1_B1_B2_A1

### Relational analysis result of IS_A2_A2_B1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -2.3815024, upper bound: 2.4000230
time: 1.58 seconds

## Relational analysis of IS_A2_A2_B1_B1_B2_A2

### Relational analysis result of IS_A2_A2_B1_B1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -2.3892668, upper bound: 2.4010396
time: 1.55 seconds

## BFS IS instance: IS_A2_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.5750360, 0.7596831, -0.2279375, 0.4104772, -0.9855133, 0.9876207
1: -0.5210941, 0.5425836, -0.2251161, 0.2422504, -0.7633445, 0.7676997
2: -0.4643506, 0.7424308, -0.2453647, 0.3261958, -0.7905463, 0.9877955
3: -0.4668015, 0.6070364, -0.1702364, 0.2920135, -0.7588149, 0.7772727
4: -0.5843288, 0.5418962, -0.2389041, 0.2109950, -0.7953238, 0.7808003
5: -0.5174262, 0.6300395, -0.2269095, 0.3014266, -0.8188528, 0.8569490
6: -0.5317135, 0.6264614, -0.2254197, 0.2903287, -0.8220422, 0.8518811
7: -0.5742255, 0.6752260, -0.2638674, 0.2760327, -0.8502583, 0.9390934
8: -0.3569139, 1.2176368, 0.3798373, 1.1268741, -1.4837880, 0.8377995
9: -0.6329191, 0.7102921, -0.3039545, 0.3880288, -1.0209479, 1.0142466

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=31, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=73, inp2_unstable=32, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_A2_B1_B2_B2_A1

### Relational analysis result of IS_A2_A2_B1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -2.3816673, upper bound: 2.4029464
time: 1.52 seconds

## Relational analysis of IS_A2_A2_B1_B2_B2_A2

### Relational analysis result of IS_A2_A2_B1_B2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -2.3894759, upper bound: 2.4039128
time: 1.83 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 4.79 seconds
IS_A2_A2_B1_B1_B2_A1, status: Status.VERIFIED, split count: 6, time: 4.79
Output dim: 8, lower bound: -2.3815024, upper bound: 2.4000230
IS_A2_A2_B1_B1_B2_A2, status: Status.VERIFIED, split count: 6, time: 4.79
Output dim: 8, lower bound: -2.3892668, upper bound: 2.4010396
IS_A2_A2_B1_B2_B2_A1, status: Status.VERIFIED, split count: 6, time: 4.79
Output dim: 8, lower bound: -2.3816673, upper bound: 2.4029464
IS_A2_A2_B1_B2_B2_A2, status: Status.VERIFIED, split count: 6, time: 4.79
Output dim: 8, lower bound: -2.3894759, upper bound: 2.4039128

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 5.25 + 70.41 = 75.66 seconds
