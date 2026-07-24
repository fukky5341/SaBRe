## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 11)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00368946


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0057215, 0.0077068, 0.0057215, 0.0077068, -0.0019853, 0.0019853)
1: (-0.0017136, 0.0030095, -0.0017136, 0.0030095, -0.0047231, 0.0047231)
2: (-0.0073460, 0.0247720, -0.0073460, 0.0247720, -0.0321180, 0.0321180)
3: (-0.0046267, -0.0018565, -0.0046267, -0.0018565, -0.0027702, 0.0027702)
4: (-0.0012160, 0.0123290, -0.0012160, 0.0123290, -0.0135449, 0.0135449)
5: (-0.0023602, 0.0007184, -0.0023602, 0.0007184, -0.0030785, 0.0030785)
6: (0.9888180, 0.9945819, 0.9888180, 0.9945819, -0.0057639, 0.0057639)
7: (-0.0157582, 0.0089347, -0.0157582, 0.0089347, -0.0246929, 0.0246929)
8: (-0.0092003, 0.0037875, -0.0092003, 0.0037875, -0.0129878, 0.0129878)
9: (-0.0148885, 0.0010365, -0.0148885, 0.0010365, -0.0159250, 0.0159250)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.06 + 3.65 = 5.71 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.0040994, upper bound: 0.0040994

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 80

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040587, upper bound: 0.0039893
time: 2.71 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040587, upper bound: 0.0040586
time: 2.25 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 5.16 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 5.16
Output dim: 6, lower bound: -0.0040587, upper bound: 0.0039893
NS_A2, status: Status.UNKNOWN, split count: 1, time: 5.16
Output dim: 6, lower bound: -0.0040587, upper bound: 0.0040586

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: 0.0057769, 0.0075913, 0.0057283, 0.0076709, -0.0018940, 0.0018630
1: -0.0012236, 0.0027857, -0.0016535, 0.0029399, -0.0041635, 0.0044392
2: -0.0055410, 0.0234257, -0.0067851, 0.0246069, -0.0301478, 0.0302108
3: -0.0045493, -0.0020177, -0.0046172, -0.0019066, -0.0026428, 0.0025995
4: -0.0007363, 0.0115468, -0.0010837, 0.0120859, -0.0128222, 0.0126305
5: -0.0022434, -0.0000043, -0.0023239, 0.0003811, -0.0026246, 0.0023196
6: 0.9899439, 0.9943677, 0.9890990, 0.9945153, -0.0045714, 0.0052687
7: -0.0149206, 0.0075188, -0.0156555, 0.0084947, -0.0234153, 0.0231743
8: -0.0066485, 0.0033439, -0.0088873, 0.0036497, -0.0102981, 0.0122313
9: -0.0140032, 0.0003012, -0.0146134, 0.0009460, -0.0149492, 0.0149145

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 96

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039759, upper bound: 0.0039477
time: 2.23 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039759, upper bound: 0.0039477
time: 2.76 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: 0.0057335, 0.0076757, 0.0057235, 0.0077013, -0.0019678, 0.0019522
1: -0.0016077, 0.0029493, -0.0016956, 0.0029988, -0.0046066, 0.0046448
2: -0.0068605, 0.0244810, -0.0072603, 0.0247223, -0.0315829, 0.0317413
3: -0.0046100, -0.0018998, -0.0046238, -0.0018641, -0.0027458, 0.0027240
4: -0.0010305, 0.0121186, -0.0011762, 0.0122918, -0.0133223, 0.0132948
5: -0.0023288, 0.0002853, -0.0023546, 0.0006170, -0.0029458, 0.0026399
6: 0.9892205, 0.9945242, 0.9889024, 0.9945717, -0.0053512, 0.0056218
7: -0.0155772, 0.0085539, -0.0157274, 0.0088674, -0.0244446, 0.0242812
8: -0.0086488, 0.0036682, -0.0091062, 0.0037665, -0.0124153, 0.0127744
9: -0.0146504, 0.0008773, -0.0148464, 0.0010093, -0.0156597, 0.0157237

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 96

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039759, upper bound: 0.0040253
time: 2.95 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039759, upper bound: 0.0040253
time: 2.73 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 7.60 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 7.60
Output dim: 6, lower bound: -0.0039759, upper bound: 0.0039477
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 7.60
Output dim: 6, lower bound: -0.0039759, upper bound: 0.0039477
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 7.60
Output dim: 6, lower bound: -0.0039759, upper bound: 0.0040253
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 7.60
Output dim: 6, lower bound: -0.0039759, upper bound: 0.0040253

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: 0.0057828, 0.0075730, 0.0057767, 0.0076048, -0.0018220, 0.0017963
1: -0.0011716, 0.0027503, -0.0012254, 0.0028119, -0.0039835, 0.0039756
2: -0.0052553, 0.0232827, -0.0057528, 0.0234304, -0.0286857, 0.0290355
3: -0.0045411, -0.0020432, -0.0045496, -0.0019988, -0.0025424, 0.0025064
4: -0.0006965, 0.0114230, -0.0007377, 0.0116386, -0.0123351, 0.0121606
5: -0.0022249, -0.0000435, -0.0022571, -0.0000030, -0.0022219, 0.0022136
6: 0.9900417, 0.9943338, 0.9899405, 0.9943928, -0.0043510, 0.0043933
7: -0.0148316, 0.0072947, -0.0149235, 0.0076850, -0.0225166, 0.0222182
8: -0.0063775, 0.0032737, -0.0066574, 0.0033960, -0.0097734, 0.0099311
9: -0.0138631, 0.0002231, -0.0141071, 0.0003037, -0.0141668, 0.0143302

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of NS_A1_B1_B1

### Relational analysis result of NS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038560, upper bound: 0.0039132
time: 1.93 seconds

## Relational analysis of NS_A1_B1_B2

### Relational analysis result of NS_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039429, upper bound: 0.0039132
time: 2.32 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: 0.0057790, 0.0075848, 0.0057381, 0.0076433, -0.0018643, 0.0018467
1: -0.0012053, 0.0027732, -0.0015668, 0.0028865, -0.0040918, 0.0043400
2: -0.0054405, 0.0233753, -0.0063541, 0.0243685, -0.0298090, 0.0297294
3: -0.0045464, -0.0020267, -0.0046035, -0.0019451, -0.0026014, 0.0025768
4: -0.0007223, 0.0115032, -0.0009992, 0.0118991, -0.0126214, 0.0125024
5: -0.0022369, -0.0000181, -0.0022960, 0.0002544, -0.0024913, 0.0022779
6: 0.9899783, 0.9943558, 0.9892976, 0.9944642, -0.0044859, 0.0050582
7: -0.0148893, 0.0074400, -0.0155072, 0.0081566, -0.0230459, 0.0229472
8: -0.0065530, 0.0033192, -0.0084356, 0.0035437, -0.0100968, 0.0117548
9: -0.0139539, 0.0002737, -0.0144020, 0.0008158, -0.0147697, 0.0146756

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039888, upper bound: 0.0038412
time: 2.76 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039429, upper bound: 0.0039132
time: 3.12 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: 0.0057391, 0.0076563, 0.0057717, 0.0076342, -0.0018951, 0.0018846
1: -0.0015577, 0.0029116, -0.0012700, 0.0028688, -0.0044265, 0.0041817
2: -0.0065571, 0.0243434, -0.0062119, 0.0235531, -0.0301102, 0.0305553
3: -0.0046021, -0.0019269, -0.0045566, -0.0019578, -0.0026443, 0.0026297
4: -0.0009922, 0.0119871, -0.0007719, 0.0118375, -0.0128297, 0.0127589
5: -0.0023092, 0.0002475, -0.0022868, 0.0000307, -0.0023398, 0.0025344
6: 0.9893148, 0.9944882, 0.9898564, 0.9944474, -0.0051326, 0.0046318
7: -0.0154916, 0.0083158, -0.0149999, 0.0080451, -0.0235367, 0.0233157
8: -0.0083880, 0.0035936, -0.0068900, 0.0035088, -0.0118968, 0.0104836
9: -0.0145015, 0.0008021, -0.0143322, 0.0003707, -0.0148722, 0.0151344

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of NS_A2_B1_B1

### Relational analysis result of NS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039303, upper bound: 0.0040252
time: 2.00 seconds

## Relational analysis of NS_A2_B1_B2

### Relational analysis result of NS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039303, upper bound: 0.0040252
time: 2.45 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: 0.0057356, 0.0076701, 0.0057333, 0.0076747, -0.0019391, 0.0019368
1: -0.0015892, 0.0029383, -0.0016092, 0.0029472, -0.0045364, 0.0045475
2: -0.0067724, 0.0244300, -0.0068442, 0.0244849, -0.0312573, 0.0312741
3: -0.0046070, -0.0019077, -0.0046102, -0.0019013, -0.0027057, 0.0027025
4: -0.0010163, 0.0120804, -0.0010316, 0.0121115, -0.0131278, 0.0131120
5: -0.0023231, 0.0002713, -0.0023277, 0.0002863, -0.0026094, 0.0025990
6: 0.9892555, 0.9945138, 0.9892179, 0.9945223, -0.0052668, 0.0052959
7: -0.0155455, 0.0084847, -0.0155796, 0.0085410, -0.0240865, 0.0240644
8: -0.0085520, 0.0036465, -0.0086562, 0.0036642, -0.0122162, 0.0123027
9: -0.0146072, 0.0008494, -0.0146424, 0.0008794, -0.0154865, 0.0154917

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of NS_A2_B2_B1

### Relational analysis result of NS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039477, upper bound: 0.0040252
time: 2.71 seconds

## Relational analysis of NS_A2_B2_B2

### Relational analysis result of NS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039477, upper bound: 0.0040253
time: 2.65 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 7.34 seconds
NS_A1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 7.34
Output dim: 6, lower bound: -0.0038560, upper bound: 0.0039132
NS_A1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 7.34
Output dim: 6, lower bound: -0.0039429, upper bound: 0.0039132
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 7.34
Output dim: 6, lower bound: -0.0039888, upper bound: 0.0038412
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 7.34
Output dim: 6, lower bound: -0.0039429, upper bound: 0.0039132
NS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 7.34
Output dim: 6, lower bound: -0.0039303, upper bound: 0.0040252
NS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 7.34
Output dim: 6, lower bound: -0.0039303, upper bound: 0.0040252
NS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 7.34
Output dim: 6, lower bound: -0.0039477, upper bound: 0.0040252
NS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 7.34
Output dim: 6, lower bound: -0.0039477, upper bound: 0.0040253

## BFS NS instance: NS_A1_B1_B1

### Backsubstitution after applying NS history:
0: 0.0057929, 0.0075364, 0.0058009, 0.0074912, -0.0016983, 0.0017355
1: -0.0010824, 0.0026795, -0.0010116, 0.0025918, -0.0036742, 0.0036911
2: -0.0046843, 0.0230376, -0.0039775, 0.0228432, -0.0275274, 0.0270152
3: -0.0045270, -0.0020942, -0.0045159, -0.0021573, -0.0023697, 0.0024217
4: -0.0006282, 0.0111755, -0.0005740, 0.0108693, -0.0114974, 0.0117495
5: -0.0021880, -0.0001108, -0.0021423, -0.0001641, -0.0020239, 0.0020315
6: 0.9902096, 0.9942660, 0.9903430, 0.9941822, -0.0039725, 0.0039231
7: -0.0146792, 0.0068468, -0.0145582, 0.0062924, -0.0209715, 0.0214049
8: -0.0059129, 0.0031334, -0.0055443, 0.0029597, -0.0088726, 0.0086777
9: -0.0135830, 0.0000893, -0.0132363, -0.0000168, -0.0135661, 0.0133256

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of NS_A1_B1_B1_B1

### Relational analysis result of NS_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038212, upper bound: 0.0039132
time: 3.35 seconds

## Relational analysis of NS_A1_B1_B1_B2

### Relational analysis result of NS_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038212, upper bound: 0.0039132
time: 2.45 seconds

## BFS NS instance: NS_A1_B1_B2

### Backsubstitution after applying NS history:
0: 0.0057834, 0.0075716, 0.0057863, 0.0075833, -0.0017999, 0.0017853
1: -0.0011663, 0.0027476, -0.0011408, 0.0027703, -0.0039367, 0.0038885
2: -0.0052341, 0.0232682, -0.0054172, 0.0231982, -0.0284323, 0.0286854
3: -0.0045403, -0.0020451, -0.0045363, -0.0020287, -0.0025115, 0.0024912
4: -0.0006925, 0.0114138, -0.0006729, 0.0114931, -0.0121856, 0.0120867
5: -0.0022236, -0.0000475, -0.0022354, -0.0000667, -0.0021568, 0.0021879
6: 0.9900517, 0.9943312, 0.9900997, 0.9943530, -0.0043013, 0.0042315
7: -0.0148226, 0.0072780, -0.0147791, 0.0074217, -0.0222443, 0.0220571
8: -0.0063500, 0.0032685, -0.0062173, 0.0033135, -0.0096635, 0.0094858
9: -0.0138526, 0.0002152, -0.0139424, 0.0001770, -0.0140296, 0.0141576

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of NS_A1_B1_B2_B1

### Relational analysis result of NS_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039018, upper bound: 0.0039132
time: 3.08 seconds

## Relational analysis of NS_A1_B1_B2_B2

### Relational analysis result of NS_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039018, upper bound: 0.0039132
time: 2.60 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.0058024, 0.0074716, 0.0057484, 0.0076079, -0.0018055, 0.0017233
1: -0.0009986, 0.0025539, -0.0014761, 0.0028179, -0.0038165, 0.0040300
2: -0.0036715, 0.0228074, -0.0058011, 0.0241193, -0.0277909, 0.0286086
3: -0.0045138, -0.0021846, -0.0045892, -0.0019945, -0.0025194, 0.0024045
4: -0.0005640, 0.0107367, -0.0009297, 0.0116595, -0.0122235, 0.0116664
5: -0.0021225, -0.0001740, -0.0022603, 0.0001860, -0.0023085, 0.0020863
6: 0.9903675, 0.9941459, 0.9894684, 0.9943985, -0.0040311, 0.0046775
7: -0.0145359, 0.0060523, -0.0153522, 0.0077228, -0.0222588, 0.0214045
8: -0.0054766, 0.0028845, -0.0079632, 0.0034079, -0.0088845, 0.0108477
9: -0.0130862, -0.0000363, -0.0141308, 0.0006798, -0.0137660, 0.0140944

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039018, upper bound: 0.0038412
time: 2.53 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039018, upper bound: 0.0038412
time: 2.79 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.0057888, 0.0075625, 0.0057387, 0.0076418, -0.0018530, 0.0018238
1: -0.0011183, 0.0027299, -0.0015615, 0.0028837, -0.0040019, 0.0042914
2: -0.0050914, 0.0231362, -0.0063313, 0.0243540, -0.0294453, 0.0294676
3: -0.0045327, -0.0020578, -0.0046027, -0.0019471, -0.0025856, 0.0025448
4: -0.0006557, 0.0113519, -0.0009951, 0.0118892, -0.0125449, 0.0123470
5: -0.0022143, -0.0000837, -0.0022946, 0.0002504, -0.0024648, 0.0022108
6: 0.9901422, 0.9943143, 0.9893075, 0.9944614, -0.0043193, 0.0050068
7: -0.0147405, 0.0071661, -0.0154982, 0.0081387, -0.0228792, 0.0226643
8: -0.0060998, 0.0032334, -0.0084080, 0.0035382, -0.0096380, 0.0116414
9: -0.0137826, 0.0001431, -0.0143908, 0.0008079, -0.0145905, 0.0145340

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039237, upper bound: 0.0039132
time: 2.73 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039237, upper bound: 0.0039132
time: 2.88 seconds

## BFS NS instance: NS_A2_B1_B1

### Backsubstitution after applying NS history:
0: 0.0057391, 0.0076563, 0.0058157, 0.0075274, -0.0017883, 0.0018406
1: -0.0015577, 0.0029116, -0.0008812, 0.0026620, -0.0042197, 0.0037929
2: -0.0065571, 0.0243434, -0.0045434, 0.0224849, -0.0290420, 0.0288868
3: -0.0046021, -0.0019269, -0.0044953, -0.0021068, -0.0024953, 0.0025684
4: -0.0009922, 0.0119871, -0.0004741, 0.0111145, -0.0121066, 0.0124612
5: -0.0023092, 0.0002475, -0.0021789, -0.0002624, -0.0020467, 0.0024264
6: 0.9893148, 0.9944882, 0.9905885, 0.9942493, -0.0049345, 0.0038997
7: -0.0154916, 0.0083158, -0.0143352, 0.0067362, -0.0222278, 0.0226511
8: -0.0083880, 0.0035936, -0.0048653, 0.0030988, -0.0114868, 0.0084589
9: -0.0145015, 0.0008021, -0.0135138, -0.0002124, -0.0142891, 0.0143160

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of NS_A2_B1_B1_A1

### Relational analysis result of NS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038975, upper bound: 0.0039111
time: 2.56 seconds

## Relational analysis of NS_A2_B1_B1_A2

### Relational analysis result of NS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038975, upper bound: 0.0039888
time: 2.37 seconds

## BFS NS instance: NS_A2_B1_B2

### Backsubstitution after applying NS history:
0: 0.0057391, 0.0076563, 0.0057811, 0.0076072, -0.0018680, 0.0018752
1: -0.0015577, 0.0029116, -0.0011868, 0.0028165, -0.0043742, 0.0040984
2: -0.0065571, 0.0243434, -0.0057897, 0.0233243, -0.0298814, 0.0301331
3: -0.0046021, -0.0019269, -0.0045435, -0.0019955, -0.0026066, 0.0026166
4: -0.0009922, 0.0119871, -0.0007081, 0.0116545, -0.0126467, 0.0126952
5: -0.0023092, 0.0002475, -0.0022595, -0.0000321, -0.0022770, 0.0025070
6: 0.9893148, 0.9944882, 0.9900132, 0.9943972, -0.0050824, 0.0044749
7: -0.0154916, 0.0083158, -0.0148575, 0.0077138, -0.0232054, 0.0231733
8: -0.0083880, 0.0035936, -0.0064564, 0.0034050, -0.0117930, 0.0100500
9: -0.0145015, 0.0008021, -0.0141251, 0.0002458, -0.0147474, 0.0149273

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of NS_A2_B1_B2_B1

### Relational analysis result of NS_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038182, upper bound: 0.0039888
time: 2.71 seconds

## Relational analysis of NS_A2_B1_B2_B2

### Relational analysis result of NS_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038975, upper bound: 0.0039888
time: 2.15 seconds

## BFS NS instance: NS_A2_B2_B1

### Backsubstitution after applying NS history:
0: 0.0057356, 0.0076701, 0.0057864, 0.0075620, -0.0018264, 0.0018837
1: -0.0015892, 0.0029383, -0.0011401, 0.0027289, -0.0043181, 0.0040784
2: -0.0067724, 0.0244300, -0.0050833, 0.0231961, -0.0299685, 0.0295133
3: -0.0046070, -0.0019077, -0.0045361, -0.0020586, -0.0025485, 0.0026284
4: -0.0010163, 0.0120804, -0.0006723, 0.0113484, -0.0123647, 0.0127527
5: -0.0023231, 0.0002713, -0.0022138, -0.0000673, -0.0022558, 0.0024851
6: 0.9892555, 0.9945138, 0.9901012, 0.9943134, -0.0050579, 0.0044127
7: -0.0155455, 0.0084847, -0.0147777, 0.0071598, -0.0227052, 0.0232625
8: -0.0085520, 0.0036465, -0.0062133, 0.0032314, -0.0117835, 0.0098598
9: -0.0146072, 0.0008494, -0.0137787, 0.0001758, -0.0147830, 0.0146281

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of NS_A2_B2_B1_A1

### Relational analysis result of NS_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039132, upper bound: 0.0039111
time: 2.56 seconds

## Relational analysis of NS_A2_B2_B1_A2

### Relational analysis result of NS_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038975, upper bound: 0.0039888
time: 2.66 seconds

## BFS NS instance: NS_A2_B2_B2

### Backsubstitution after applying NS history:
0: 0.0057356, 0.0076701, 0.0057430, 0.0076496, -0.0019140, 0.0019270
1: -0.0015892, 0.0029383, -0.0015230, 0.0028987, -0.0044879, 0.0044614
2: -0.0067724, 0.0244300, -0.0064526, 0.0242482, -0.0310206, 0.0308826
3: -0.0046070, -0.0019077, -0.0045966, -0.0019363, -0.0026708, 0.0026889
4: -0.0010163, 0.0120804, -0.0009656, 0.0119418, -0.0129581, 0.0130460
5: -0.0023231, 0.0002713, -0.0023024, 0.0002214, -0.0025445, 0.0025737
6: 0.9892555, 0.9945138, 0.9893801, 0.9944759, -0.0052204, 0.0051337
7: -0.0155455, 0.0084847, -0.0154324, 0.0082339, -0.0237793, 0.0239171
8: -0.0085520, 0.0036465, -0.0082075, 0.0035680, -0.0121200, 0.0118541
9: -0.0146072, 0.0008494, -0.0144503, 0.0007501, -0.0153573, 0.0152997

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of NS_A2_B2_B2_B1

### Relational analysis result of NS_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038182, upper bound: 0.0039888
time: 2.89 seconds

## Relational analysis of NS_A2_B2_B2_B2

### Relational analysis result of NS_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039132, upper bound: 0.0039888
time: 2.80 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 7.57 seconds
NS_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 7.57
Output dim: 6, lower bound: -0.0038212, upper bound: 0.0039132
NS_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 7.57
Output dim: 6, lower bound: -0.0038212, upper bound: 0.0039132
NS_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 7.57
Output dim: 6, lower bound: -0.0039018, upper bound: 0.0039132
NS_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 7.57
Output dim: 6, lower bound: -0.0039018, upper bound: 0.0039132
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 7.57
Output dim: 6, lower bound: -0.0039018, upper bound: 0.0038412
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 7.57
Output dim: 6, lower bound: -0.0039018, upper bound: 0.0038412
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 7.57
Output dim: 6, lower bound: -0.0039237, upper bound: 0.0039132
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 7.57
Output dim: 6, lower bound: -0.0039237, upper bound: 0.0039132
NS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 7.57
Output dim: 6, lower bound: -0.0038975, upper bound: 0.0039111
NS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 7.57
Output dim: 6, lower bound: -0.0038975, upper bound: 0.0039888
NS_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 7.57
Output dim: 6, lower bound: -0.0038182, upper bound: 0.0039888
NS_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 7.57
Output dim: 6, lower bound: -0.0038975, upper bound: 0.0039888
NS_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 7.57
Output dim: 6, lower bound: -0.0039132, upper bound: 0.0039111
NS_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 7.57
Output dim: 6, lower bound: -0.0038975, upper bound: 0.0039888
NS_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 7.57
Output dim: 6, lower bound: -0.0038182, upper bound: 0.0039888
NS_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 7.57
Output dim: 6, lower bound: -0.0039132, upper bound: 0.0039888

## BFS NS instance: NS_A1_B1_B1_B1

### Backsubstitution after applying NS history:
0: 0.0057929, 0.0075364, 0.0058420, 0.0074122, -0.0016193, 0.0016945
1: -0.0010824, 0.0026795, -0.0006486, 0.0024388, -0.0035212, 0.0033281
2: -0.0046843, 0.0230376, -0.0027430, 0.0218458, -0.0265301, 0.0257806
3: -0.0045270, -0.0020942, -0.0044586, -0.0022676, -0.0022595, 0.0023644
4: -0.0006282, 0.0111755, -0.0002960, 0.0103343, -0.0109625, 0.0114715
5: -0.0021880, -0.0001108, -0.0020624, -0.0004378, -0.0017502, 0.0019516
6: 0.9902096, 0.9942660, 0.9910264, 0.9940357, -0.0038261, 0.0032396
7: -0.0146792, 0.0068468, -0.0139376, 0.0053240, -0.0200031, 0.0207844
8: -0.0059129, 0.0031334, -0.0036539, 0.0026563, -0.0085693, 0.0067873
9: -0.0135830, 0.0000893, -0.0126308, -0.0005612, -0.0130217, 0.0127201

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A1_B1_B1_B1_A1

### Relational analysis result of NS_A1_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036736, upper bound: 0.0037612
time: 2.54 seconds

## Relational analysis of NS_A1_B1_B1_B1_A2

### Relational analysis result of NS_A1_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036460, upper bound: 0.0037679
time: 2.64 seconds

## BFS NS instance: NS_A1_B1_B1_B2

### Backsubstitution after applying NS history:
0: 0.0057929, 0.0075364, 0.0058054, 0.0074950, -0.0017021, 0.0017310
1: -0.0010824, 0.0026795, -0.0009715, 0.0025991, -0.0036816, 0.0036510
2: -0.0046843, 0.0230376, -0.0040365, 0.0227329, -0.0274172, 0.0270741
3: -0.0045270, -0.0020942, -0.0045095, -0.0021520, -0.0023750, 0.0024153
4: -0.0006282, 0.0111755, -0.0005432, 0.0108948, -0.0115230, 0.0117187
5: -0.0021880, -0.0001108, -0.0021461, -0.0001944, -0.0019936, 0.0020353
6: 0.9902096, 0.9942660, 0.9904186, 0.9941892, -0.0039796, 0.0038475
7: -0.0146792, 0.0068468, -0.0144896, 0.0063386, -0.0210178, 0.0213363
8: -0.0059129, 0.0031334, -0.0053353, 0.0029742, -0.0088871, 0.0084687
9: -0.0135830, 0.0000893, -0.0132652, -0.0000770, -0.0135059, 0.0133546

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A1_B1_B1_B2_A1

### Relational analysis result of NS_A1_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036736, upper bound: 0.0037612
time: 2.91 seconds

## Relational analysis of NS_A1_B1_B1_B2_A2

### Relational analysis result of NS_A1_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036460, upper bound: 0.0037679
time: 2.64 seconds

## BFS NS instance: NS_A1_B1_B2_B1

### Backsubstitution after applying NS history:
0: 0.0057834, 0.0075716, 0.0058251, 0.0075060, -0.0017226, 0.0017465
1: -0.0011663, 0.0027476, -0.0007975, 0.0026204, -0.0037868, 0.0035451
2: -0.0052341, 0.0232682, -0.0042082, 0.0222547, -0.0274888, 0.0274764
3: -0.0045403, -0.0020451, -0.0044821, -0.0021367, -0.0024036, 0.0024370
4: -0.0006925, 0.0114138, -0.0004099, 0.0109692, -0.0116617, 0.0118237
5: -0.0022236, -0.0000475, -0.0021572, -0.0003256, -0.0018979, 0.0021097
6: 0.9900517, 0.9943312, 0.9907463, 0.9942095, -0.0041578, 0.0035850
7: -0.0148226, 0.0072780, -0.0141920, 0.0064733, -0.0212960, 0.0214701
8: -0.0063500, 0.0032685, -0.0044290, 0.0030164, -0.0093664, 0.0076975
9: -0.0138526, 0.0002152, -0.0133495, -0.0003381, -0.0135146, 0.0135647

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of NS_A1_B1_B2_B1_A1

### Relational analysis result of NS_A1_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038212, upper bound: 0.0038412
time: 2.35 seconds

## Relational analysis of NS_A1_B1_B2_B1_A2

### Relational analysis result of NS_A1_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038212, upper bound: 0.0039132
time: 2.41 seconds

## BFS NS instance: NS_A1_B1_B2_B2

### Backsubstitution after applying NS history:
0: 0.0057834, 0.0075716, 0.0057908, 0.0075845, -0.0018011, 0.0017808
1: -0.0011663, 0.0027476, -0.0011013, 0.0027726, -0.0039389, 0.0038489
2: -0.0052341, 0.0232682, -0.0054356, 0.0230895, -0.0283236, 0.0287038
3: -0.0045403, -0.0020451, -0.0045300, -0.0020271, -0.0025132, 0.0024849
4: -0.0006925, 0.0114138, -0.0006426, 0.0115011, -0.0121935, 0.0120564
5: -0.0022236, -0.0000475, -0.0022366, -0.0000966, -0.0021270, 0.0021891
6: 0.9900517, 0.9943312, 0.9901741, 0.9943551, -0.0043035, 0.0041571
7: -0.0148226, 0.0072780, -0.0147114, 0.0074361, -0.0222587, 0.0219895
8: -0.0063500, 0.0032685, -0.0060113, 0.0033180, -0.0096680, 0.0092798
9: -0.0138526, 0.0002152, -0.0139515, 0.0001176, -0.0139703, 0.0141667

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of NS_A1_B1_B2_B2_A1

### Relational analysis result of NS_A1_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038212, upper bound: 0.0038412
time: 2.27 seconds

## Relational analysis of NS_A1_B1_B2_B2_A2

### Relational analysis result of NS_A1_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038212, upper bound: 0.0039132
time: 2.43 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.0058024, 0.0074716, 0.0057964, 0.0075260, -0.0017236, 0.0016752
1: -0.0009986, 0.0025539, -0.0010511, 0.0026592, -0.0036579, 0.0036050
2: -0.0036715, 0.0228074, -0.0045213, 0.0229517, -0.0266233, 0.0273287
3: -0.0045138, -0.0021846, -0.0045221, -0.0021088, -0.0024051, 0.0023375
4: -0.0005640, 0.0107367, -0.0006042, 0.0111049, -0.0116689, 0.0113409
5: -0.0021225, -0.0001740, -0.0021775, -0.0001344, -0.0019881, 0.0020035
6: 0.9903675, 0.9941459, 0.9902686, 0.9942467, -0.0038792, 0.0038772
7: -0.0145359, 0.0060523, -0.0146257, 0.0067189, -0.0212548, 0.0206781
8: -0.0054766, 0.0028845, -0.0057501, 0.0030933, -0.0085699, 0.0086346
9: -0.0130862, -0.0000363, -0.0135030, 0.0000424, -0.0131287, 0.0134667

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of NS_A1_B2_A1_B1_B1

### Relational analysis result of NS_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037614, upper bound: 0.0036987
time: 2.57 seconds

## Relational analysis of NS_A1_B2_A1_B1_B2

### Relational analysis result of NS_A1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037690, upper bound: 0.0036690
time: 2.77 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.0058024, 0.0074716, 0.0057534, 0.0076143, -0.0018120, 0.0017182
1: -0.0009986, 0.0025539, -0.0014316, 0.0028304, -0.0038290, 0.0039855
2: -0.0036715, 0.0228074, -0.0059015, 0.0239970, -0.0276685, 0.0287090
3: -0.0045138, -0.0021846, -0.0045822, -0.0019855, -0.0025283, 0.0023975
4: -0.0005640, 0.0107367, -0.0008956, 0.0117030, -0.0122670, 0.0116323
5: -0.0021225, -0.0001740, -0.0022667, 0.0001525, -0.0022750, 0.0020928
6: 0.9903675, 0.9941459, 0.9895521, 0.9944105, -0.0040430, 0.0045937
7: -0.0145359, 0.0060523, -0.0152761, 0.0078016, -0.0223375, 0.0213284
8: -0.0054766, 0.0028845, -0.0077314, 0.0034325, -0.0089091, 0.0106159
9: -0.0130862, -0.0000363, -0.0141800, 0.0006130, -0.0136992, 0.0141437

Time for backsubstitution: 2.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of NS_A1_B2_A1_B2_B1

### Relational analysis result of NS_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037614, upper bound: 0.0036987
time: 2.88 seconds

## Relational analysis of NS_A1_B2_A1_B2_B2

### Relational analysis result of NS_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037690, upper bound: 0.0036691
time: 2.70 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.0057888, 0.0075625, 0.0057870, 0.0075606, -0.0017717, 0.0017755
1: -0.0011183, 0.0027299, -0.0011348, 0.0027262, -0.0038445, 0.0038647
2: -0.0050914, 0.0231362, -0.0050614, 0.0231816, -0.0282730, 0.0281976
3: -0.0045327, -0.0020578, -0.0045353, -0.0020605, -0.0024722, 0.0024775
4: -0.0006557, 0.0113519, -0.0006683, 0.0113389, -0.0119946, 0.0120202
5: -0.0022143, -0.0000837, -0.0022124, -0.0000713, -0.0021431, 0.0021287
6: 0.9901422, 0.9943143, 0.9901111, 0.9943108, -0.0041686, 0.0042033
7: -0.0147405, 0.0071661, -0.0147687, 0.0071426, -0.0218831, 0.0219348
8: -0.0060998, 0.0032334, -0.0061858, 0.0032261, -0.0093259, 0.0094192
9: -0.0137826, 0.0001431, -0.0137679, 0.0001679, -0.0139505, 0.0139111

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of NS_A1_B2_A2_B1_B1

### Relational analysis result of NS_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038211, upper bound: 0.0039132
time: 2.52 seconds

## Relational analysis of NS_A1_B2_A2_B1_B2

### Relational analysis result of NS_A1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038518, upper bound: 0.0039132
time: 2.14 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.0057888, 0.0075625, 0.0057436, 0.0076481, -0.0018593, 0.0018188
1: -0.0011183, 0.0027299, -0.0015177, 0.0028958, -0.0040141, 0.0042476
2: -0.0050914, 0.0231362, -0.0064296, 0.0242336, -0.0293249, 0.0295658
3: -0.0045327, -0.0020578, -0.0045957, -0.0019383, -0.0025944, 0.0025379
4: -0.0006557, 0.0113519, -0.0009616, 0.0119318, -0.0125875, 0.0123135
5: -0.0022143, -0.0000837, -0.0023009, 0.0002174, -0.0024317, 0.0022172
6: 0.9901422, 0.9943143, 0.9893900, 0.9944730, -0.0043309, 0.0049243
7: -0.0147405, 0.0071661, -0.0154233, 0.0082158, -0.0229563, 0.0225893
8: -0.0060998, 0.0032334, -0.0081798, 0.0035623, -0.0096621, 0.0114132
9: -0.0137826, 0.0001431, -0.0144390, 0.0007422, -0.0145248, 0.0145821

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of NS_A1_B2_A2_B2_B1

### Relational analysis result of NS_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038211, upper bound: 0.0039132
time: 2.75 seconds

## Relational analysis of NS_A1_B2_A2_B2_B2

### Relational analysis result of NS_A1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038518, upper bound: 0.0039132
time: 2.13 seconds

## BFS NS instance: NS_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: 0.0057606, 0.0075450, 0.0058254, 0.0074905, -0.0017299, 0.0017196
1: -0.0013678, 0.0026961, -0.0007951, 0.0025906, -0.0039584, 0.0034912
2: -0.0048187, 0.0238218, -0.0039673, 0.0222482, -0.0270669, 0.0277891
3: -0.0045721, -0.0020822, -0.0044817, -0.0021582, -0.0024139, 0.0023995
4: -0.0008468, 0.0112337, -0.0004081, 0.0108648, -0.0117116, 0.0116419
5: -0.0021967, 0.0001044, -0.0021416, -0.0003274, -0.0018693, 0.0022460
6: 0.9896723, 0.9942820, 0.9907507, 0.9941810, -0.0045087, 0.0035313
7: -0.0151671, 0.0069522, -0.0141880, 0.0062843, -0.0214514, 0.0211402
8: -0.0073994, 0.0031664, -0.0044166, 0.0029572, -0.0103566, 0.0075830
9: -0.0136489, 0.0005174, -0.0132313, -0.0003416, -0.0133073, 0.0137487

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of NS_A2_B1_B1_A1_B1

### Relational analysis result of NS_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037473, upper bound: 0.0037971
time: 2.60 seconds

## Relational analysis of NS_A2_B1_B1_A1_B2

### Relational analysis result of NS_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037510, upper bound: 0.0037501
time: 2.89 seconds

## BFS NS instance: NS_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: 0.0057491, 0.0076331, 0.0058162, 0.0075261, -0.0017770, 0.0018169
1: -0.0014699, 0.0028668, -0.0008761, 0.0026594, -0.0041293, 0.0037429
2: -0.0061952, 0.0241023, -0.0045224, 0.0224709, -0.0286661, 0.0286247
3: -0.0045882, -0.0019592, -0.0044945, -0.0021087, -0.0024796, 0.0025352
4: -0.0009250, 0.0118303, -0.0004702, 0.0111054, -0.0120303, 0.0123005
5: -0.0022857, 0.0001814, -0.0021775, -0.0002663, -0.0020194, 0.0023589
6: 0.9894801, 0.9944453, 0.9905981, 0.9942469, -0.0047668, 0.0038472
7: -0.0153416, 0.0080320, -0.0143265, 0.0067198, -0.0220614, 0.0223585
8: -0.0079309, 0.0035047, -0.0048387, 0.0030936, -0.0110245, 0.0083434
9: -0.0143240, 0.0006705, -0.0135036, -0.0002200, -0.0141040, 0.0141741

Time for backsubstitution: 2.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of NS_A2_B1_B1_A2_B1

### Relational analysis result of NS_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038182, upper bound: 0.0039888
time: 2.75 seconds

## Relational analysis of NS_A2_B1_B1_A2_B2

### Relational analysis result of NS_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038182, upper bound: 0.0039888
time: 3.25 seconds

## BFS NS instance: NS_A2_B1_B2_B1

### Backsubstitution after applying NS history:
0: 0.0057495, 0.0076210, 0.0058054, 0.0074950, -0.0017454, 0.0018156
1: -0.0014659, 0.0028433, -0.0009715, 0.0025991, -0.0040650, 0.0038148
2: -0.0060059, 0.0240912, -0.0040365, 0.0227329, -0.0287387, 0.0281277
3: -0.0045876, -0.0019762, -0.0045095, -0.0021520, -0.0024355, 0.0025334
4: -0.0009219, 0.0117482, -0.0005432, 0.0108948, -0.0118167, 0.0122914
5: -0.0022735, 0.0001783, -0.0021461, -0.0001944, -0.0020791, 0.0023244
6: 0.9894876, 0.9944229, 0.9904186, 0.9941892, -0.0047016, 0.0040043
7: -0.0153347, 0.0078834, -0.0144896, 0.0063386, -0.0216733, 0.0223730
8: -0.0079099, 0.0034582, -0.0053353, 0.0029742, -0.0108841, 0.0087935
9: -0.0142312, 0.0006644, -0.0132652, -0.0000770, -0.0141542, 0.0139297

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A2_B1_B2_B1_A1

### Relational analysis result of NS_A2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036722, upper bound: 0.0038232
time: 2.33 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2

### Relational analysis result of NS_A2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036486, upper bound: 0.0038502
time: 2.85 seconds

## BFS NS instance: NS_A2_B1_B2_B2

### Backsubstitution after applying NS history:
0: 0.0057397, 0.0076548, 0.0057908, 0.0075845, -0.0018448, 0.0018641
1: -0.0015523, 0.0029088, -0.0011013, 0.0027726, -0.0043249, 0.0040101
2: -0.0065343, 0.0243288, -0.0054356, 0.0230895, -0.0296237, 0.0297644
3: -0.0046012, -0.0019290, -0.0045300, -0.0020271, -0.0025741, 0.0026011
4: -0.0009881, 0.0119772, -0.0006426, 0.0115011, -0.0124892, 0.0126198
5: -0.0023077, 0.0002435, -0.0022366, -0.0000966, -0.0022111, 0.0024801
6: 0.9893248, 0.9944855, 0.9901741, 0.9943551, -0.0050303, 0.0043114
7: -0.0154825, 0.0082979, -0.0147114, 0.0074361, -0.0229186, 0.0230093
8: -0.0083603, 0.0035880, -0.0060113, 0.0033180, -0.0116783, 0.0095993
9: -0.0144903, 0.0007941, -0.0139515, 0.0001176, -0.0146080, 0.0147456

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of NS_A2_B1_B2_B2_A1

### Relational analysis result of NS_A2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038975, upper bound: 0.0039111
time: 3.24 seconds

## Relational analysis of NS_A2_B1_B2_B2_A2

### Relational analysis result of NS_A2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038975, upper bound: 0.0039888
time: 5.90 seconds

## BFS NS instance: NS_A2_B2_B1_A1

### Backsubstitution after applying NS history:
0: 0.0057572, 0.0075594, 0.0057964, 0.0075260, -0.0017688, 0.0017630
1: -0.0013975, 0.0027240, -0.0010511, 0.0026592, -0.0040568, 0.0037751
2: -0.0050434, 0.0239035, -0.0045213, 0.0229517, -0.0279952, 0.0284248
3: -0.0045768, -0.0020621, -0.0045221, -0.0021088, -0.0024680, 0.0024600
4: -0.0008695, 0.0113312, -0.0006042, 0.0111049, -0.0119744, 0.0119354
5: -0.0022112, 0.0001268, -0.0021775, -0.0001344, -0.0020769, 0.0023043
6: 0.9896164, 0.9943086, 0.9902686, 0.9942467, -0.0046303, 0.0040399
7: -0.0152179, 0.0071285, -0.0146257, 0.0067189, -0.0219368, 0.0217542
8: -0.0075541, 0.0032216, -0.0057501, 0.0030933, -0.0106474, 0.0089718
9: -0.0137591, 0.0005620, -0.0135030, 0.0000424, -0.0138016, 0.0140650

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A2_B2_B1_A1_A1

### Relational analysis result of NS_A2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038062, upper bound: 0.0037444
time: 2.79 seconds

## Relational analysis of NS_A2_B2_B1_A1_A2

### Relational analysis result of NS_A2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037510, upper bound: 0.0037501
time: 2.49 seconds

## BFS NS instance: NS_A2_B2_B1_A2

### Backsubstitution after applying NS history:
0: 0.0057455, 0.0076467, 0.0057870, 0.0075606, -0.0018151, 0.0018597
1: -0.0015011, 0.0028930, -0.0011348, 0.0027262, -0.0042273, 0.0040278
2: -0.0064065, 0.0241880, -0.0050614, 0.0231816, -0.0295881, 0.0292494
3: -0.0045931, -0.0019404, -0.0045353, -0.0020605, -0.0025326, 0.0025949
4: -0.0009488, 0.0119218, -0.0006683, 0.0113389, -0.0122878, 0.0125901
5: -0.0022994, 0.0002049, -0.0022124, -0.0000713, -0.0022281, 0.0024173
6: 0.9894214, 0.9944704, 0.9901111, 0.9943108, -0.0048894, 0.0043594
7: -0.0153949, 0.0081977, -0.0147687, 0.0071426, -0.0225375, 0.0229664
8: -0.0080934, 0.0035566, -0.0061858, 0.0032261, -0.0113194, 0.0097424
9: -0.0144277, 0.0007173, -0.0137679, 0.0001679, -0.0145956, 0.0144852

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of NS_A2_B2_B1_A2_B1

### Relational analysis result of NS_A2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038412, upper bound: 0.0039888
time: 2.63 seconds

## Relational analysis of NS_A2_B2_B1_A2_B2

### Relational analysis result of NS_A2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038412, upper bound: 0.0039888
time: 2.23 seconds

## BFS NS instance: NS_A2_B2_B2_B1

### Backsubstitution after applying NS history:
0: 0.0057460, 0.0076348, 0.0057645, 0.0075392, -0.0017933, 0.0018704
1: -0.0014972, 0.0028701, -0.0013337, 0.0026849, -0.0041821, 0.0042037
2: -0.0062217, 0.0241774, -0.0047281, 0.0237279, -0.0299496, 0.0289055
3: -0.0045925, -0.0019569, -0.0045667, -0.0020903, -0.0025022, 0.0026098
4: -0.0009459, 0.0118417, -0.0008206, 0.0111945, -0.0121404, 0.0126623
5: -0.0022875, 0.0002020, -0.0021908, 0.0000786, -0.0023661, 0.0023928
6: 0.9894286, 0.9944484, 0.9897366, 0.9942713, -0.0048427, 0.0047119
7: -0.0153883, 0.0080527, -0.0151087, 0.0068811, -0.0222695, 0.0231614
8: -0.0080733, 0.0035112, -0.0072214, 0.0031442, -0.0112175, 0.0107326
9: -0.0143370, 0.0007115, -0.0136044, 0.0004662, -0.0148032, 0.0143160

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A2_B2_B2_B1_A1

### Relational analysis result of NS_A2_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036994, upper bound: 0.0038232
time: 2.27 seconds

## Relational analysis of NS_A2_B2_B2_B1_A2

### Relational analysis result of NS_A2_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036486, upper bound: 0.0038502
time: 2.02 seconds

## BFS NS instance: NS_A2_B2_B2_B2

### Backsubstitution after applying NS history:
0: 0.0057362, 0.0076686, 0.0057530, 0.0076257, -0.0018896, 0.0019156
1: -0.0015838, 0.0029355, -0.0014353, 0.0028524, -0.0044363, 0.0043708
2: -0.0067496, 0.0244153, -0.0060796, 0.0240072, -0.0307568, 0.0304949
3: -0.0046062, -0.0019097, -0.0045827, -0.0019696, -0.0026366, 0.0026730
4: -0.0010122, 0.0120705, -0.0008985, 0.0117802, -0.0127924, 0.0129689
5: -0.0023216, 0.0002672, -0.0022783, 0.0001553, -0.0024769, 0.0025455
6: 0.9892656, 0.9945111, 0.9895453, 0.9944316, -0.0051661, 0.0049658
7: -0.0155363, 0.0084668, -0.0152824, 0.0079413, -0.0234776, 0.0237492
8: -0.0085242, 0.0036409, -0.0077507, 0.0034763, -0.0120005, 0.0113917
9: -0.0145960, 0.0008413, -0.0142673, 0.0006186, -0.0152146, 0.0151087

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of NS_A2_B2_B2_B2_A1

### Relational analysis result of NS_A2_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038975, upper bound: 0.0039111
time: 2.59 seconds

## Relational analysis of NS_A2_B2_B2_B2_A2

### Relational analysis result of NS_A2_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039132, upper bound: 0.0039888
time: 2.79 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 7.26 seconds
NS_A1_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.26
Output dim: 6, lower bound: -0.0036736, upper bound: 0.0037612
NS_A1_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.26
Output dim: 6, lower bound: -0.0036460, upper bound: 0.0037679
NS_A1_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.26
Output dim: 6, lower bound: -0.0036736, upper bound: 0.0037612
NS_A1_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.26
Output dim: 6, lower bound: -0.0036460, upper bound: 0.0037679
NS_A1_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.26
Output dim: 6, lower bound: -0.0038212, upper bound: 0.0038412
NS_A1_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.26
Output dim: 6, lower bound: -0.0038212, upper bound: 0.0039132
NS_A1_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.26
Output dim: 6, lower bound: -0.0038212, upper bound: 0.0038412
NS_A1_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.26
Output dim: 6, lower bound: -0.0038212, upper bound: 0.0039132
NS_A1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 7.26
Output dim: 6, lower bound: -0.0037614, upper bound: 0.0036987
NS_A1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 7.26
Output dim: 6, lower bound: -0.0037690, upper bound: 0.0036690
NS_A1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 7.26
Output dim: 6, lower bound: -0.0037614, upper bound: 0.0036987
NS_A1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 7.26
Output dim: 6, lower bound: -0.0037690, upper bound: 0.0036691
NS_A1_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 7.26
Output dim: 6, lower bound: -0.0038211, upper bound: 0.0039132
NS_A1_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 7.26
Output dim: 6, lower bound: -0.0038518, upper bound: 0.0039132
NS_A1_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 7.26
Output dim: 6, lower bound: -0.0038211, upper bound: 0.0039132
NS_A1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 7.26
Output dim: 6, lower bound: -0.0038518, upper bound: 0.0039132
NS_A2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 7.26
Output dim: 6, lower bound: -0.0037473, upper bound: 0.0037971
NS_A2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 7.26
Output dim: 6, lower bound: -0.0037510, upper bound: 0.0037501
NS_A2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 7.26
Output dim: 6, lower bound: -0.0038182, upper bound: 0.0039888
NS_A2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 7.26
Output dim: 6, lower bound: -0.0038182, upper bound: 0.0039888
NS_A2_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.26
Output dim: 6, lower bound: -0.0036722, upper bound: 0.0038232
NS_A2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.26
Output dim: 6, lower bound: -0.0036486, upper bound: 0.0038502
NS_A2_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.26
Output dim: 6, lower bound: -0.0038975, upper bound: 0.0039111
NS_A2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.26
Output dim: 6, lower bound: -0.0038975, upper bound: 0.0039888
NS_A2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 7.26
Output dim: 6, lower bound: -0.0038062, upper bound: 0.0037444
NS_A2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 7.26
Output dim: 6, lower bound: -0.0037510, upper bound: 0.0037501
NS_A2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 7.26
Output dim: 6, lower bound: -0.0038412, upper bound: 0.0039888
NS_A2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 7.26
Output dim: 6, lower bound: -0.0038412, upper bound: 0.0039888
NS_A2_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.26
Output dim: 6, lower bound: -0.0036994, upper bound: 0.0038232
NS_A2_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.26
Output dim: 6, lower bound: -0.0036486, upper bound: 0.0038502
NS_A2_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.26
Output dim: 6, lower bound: -0.0038975, upper bound: 0.0039111
NS_A2_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.26
Output dim: 6, lower bound: -0.0039132, upper bound: 0.0039888

## BFS NS instance: NS_A1_B1_B1_B1_A1

### Backsubstitution after applying NS history:
0: 0.0058105, 0.0074932, 0.0058434, 0.0074089, -0.0015985, 0.0016499
1: -0.0009270, 0.0025958, -0.0006360, 0.0024325, -0.0033595, 0.0032318
2: -0.0040099, 0.0226106, -0.0026927, 0.0218110, -0.0258209, 0.0253033
3: -0.0045025, -0.0021544, -0.0044566, -0.0022721, -0.0022304, 0.0023021
4: -0.0005092, 0.0108833, -0.0002863, 0.0103125, -0.0108217, 0.0111695
5: -0.0021444, -0.0002280, -0.0020592, -0.0004474, -0.0016970, 0.0018312
6: 0.9905024, 0.9941860, 0.9910504, 0.9940298, -0.0035275, 0.0031356
7: -0.0144135, 0.0063177, -0.0139160, 0.0052845, -0.0196980, 0.0202337
8: -0.0051036, 0.0029676, -0.0035880, 0.0026439, -0.0077476, 0.0065557
9: -0.0132522, -0.0001437, -0.0126061, -0.0005802, -0.0126719, 0.0124624

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of NS_A1_B1_B1_B1_A1_A1

### Relational analysis result of NS_A1_B1_B1_B1_A1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0034128, upper bound: 0.0033857
time: 2.02 seconds

## Relational analysis of NS_A1_B1_B1_B1_A1_A2

### Relational analysis result of NS_A1_B1_B1_B1_A1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0032687, upper bound: 0.0033784
time: 2.13 seconds

## BFS NS instance: NS_A1_B1_B1_B1_A2

### Backsubstitution after applying NS history:
0: 0.0057498, 0.0074972, 0.0058509, 0.0073957, -0.0014644, 0.0016463
1: -0.0014631, 0.0026035, -0.0005852, 0.0024069, -0.0038700, 0.0028236
2: -0.0040719, 0.0240835, -0.0024858, 0.0216484, -0.0227745, 0.0265693
3: -0.0045871, -0.0021489, -0.0044461, -0.0022905, -0.0020434, 0.0022972
4: -0.0009197, 0.0109101, -0.0002355, 0.0102228, -0.0099142, 0.0111456
5: -0.0021484, 0.0001762, -0.0020458, -0.0004846, -0.0014732, 0.0022220
6: 0.9894930, 0.9941934, 0.9911419, 0.9940053, -0.0045123, 0.0027020
7: -0.0153299, 0.0063664, -0.0138091, 0.0051222, -0.0204521, 0.0178648
8: -0.0078954, 0.0029829, -0.0033379, 0.0025931, -0.0104885, 0.0055969
9: -0.0132826, 0.0006603, -0.0125046, -0.0006671, -0.0111707, 0.0131649

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of NS_A1_B1_B1_B1_A2_A1

### Relational analysis result of NS_A1_B1_B1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0033714, upper bound: 0.0033827
time: 3.82 seconds

## Relational analysis of NS_A1_B1_B1_B1_A2_A2

### Relational analysis result of NS_A1_B1_B1_B1_A2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0032362, upper bound: 0.0033766
time: 2.44 seconds

## BFS NS instance: NS_A1_B1_B1_B2_A1

### Backsubstitution after applying NS history:
0: 0.0058105, 0.0074932, 0.0058069, 0.0074916, -0.0016811, 0.0016864
1: -0.0009270, 0.0025958, -0.0009587, 0.0025926, -0.0035196, 0.0035546
2: -0.0040099, 0.0226106, -0.0039838, 0.0226979, -0.0267077, 0.0265944
3: -0.0045025, -0.0021544, -0.0045075, -0.0021568, -0.0023457, 0.0023531
4: -0.0005092, 0.0108833, -0.0005335, 0.0108719, -0.0113811, 0.0114167
5: -0.0021444, -0.0002280, -0.0021427, -0.0002040, -0.0019404, 0.0019147
6: 0.9905024, 0.9941860, 0.9904426, 0.9941829, -0.0036806, 0.0037434
7: -0.0144135, 0.0063177, -0.0144678, 0.0062973, -0.0207108, 0.0207855
8: -0.0051036, 0.0029676, -0.0052689, 0.0029612, -0.0080649, 0.0082366
9: -0.0132522, -0.0001437, -0.0132394, -0.0000961, -0.0131560, 0.0130956

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of NS_A1_B1_B1_B2_A1_A1

### Relational analysis result of NS_A1_B1_B1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0035001, upper bound: 0.0033863
time: 2.64 seconds

## Relational analysis of NS_A1_B1_B1_B2_A1_A2

### Relational analysis result of NS_A1_B1_B1_B2_A1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0033459, upper bound: 0.0033791
time: 5.55 seconds

## BFS NS instance: NS_A1_B1_B1_B2_A2

### Backsubstitution after applying NS history:
0: 0.0057498, 0.0074972, 0.0058142, 0.0074793, -0.0017295, 0.0016830
1: -0.0014631, 0.0026035, -0.0008941, 0.0025688, -0.0040319, 0.0034977
2: -0.0040719, 0.0240835, -0.0037919, 0.0225204, -0.0265923, 0.0278754
3: -0.0045871, -0.0021489, -0.0044973, -0.0021739, -0.0024132, 0.0023484
4: -0.0009197, 0.0109101, -0.0004840, 0.0107888, -0.0117085, 0.0113941
5: -0.0021484, 0.0001762, -0.0021303, -0.0002527, -0.0018957, 0.0023065
6: 0.9894930, 0.9941934, 0.9905642, 0.9941601, -0.0046671, 0.0036291
7: -0.0153299, 0.0063664, -0.0143573, 0.0061467, -0.0214766, 0.0207237
8: -0.0078954, 0.0029829, -0.0049325, 0.0029141, -0.0108095, 0.0079154
9: -0.0132826, 0.0006603, -0.0131452, -0.0001930, -0.0130896, 0.0138055

Time for backsubstitution: 2.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of NS_A1_B1_B1_B2_A2_B1

### Relational analysis result of NS_A1_B1_B1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0032600, upper bound: 0.0035432
time: 3.30 seconds

## Relational analysis of NS_A1_B1_B1_B2_A2_B2

### Relational analysis result of NS_A1_B1_B1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0033036, upper bound: 0.0033767
time: 2.64 seconds

## BFS NS instance: NS_A1_B1_B2_B1_A1

### Backsubstitution after applying NS history:
0: 0.0058060, 0.0074591, 0.0058251, 0.0075060, -0.0017000, 0.0016340
1: -0.0009667, 0.0025297, -0.0007975, 0.0026204, -0.0035872, 0.0033271
2: -0.0034762, 0.0227198, -0.0042082, 0.0222547, -0.0257309, 0.0269281
3: -0.0045088, -0.0022021, -0.0044821, -0.0021367, -0.0023721, 0.0022800
4: -0.0005396, 0.0106520, -0.0004099, 0.0109692, -0.0115088, 0.0110619
5: -0.0021099, -0.0001980, -0.0021572, -0.0003256, -0.0017842, 0.0019592
6: 0.9904276, 0.9941227, 0.9907463, 0.9942095, -0.0037820, 0.0033764
7: -0.0144814, 0.0058991, -0.0141920, 0.0064733, -0.0209548, 0.0200912
8: -0.0053106, 0.0028365, -0.0044290, 0.0030164, -0.0083270, 0.0072655
9: -0.0129904, -0.0000841, -0.0133495, -0.0003381, -0.0126524, 0.0132653

Time for backsubstitution: 2.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of NS_A1_B1_B2_B1_A1_B1

### Relational analysis result of NS_A1_B1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036594, upper bound: 0.0037068
time: 2.93 seconds

## Relational analysis of NS_A1_B1_B2_B1_A1_B2

### Relational analysis result of NS_A1_B1_B2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0036460, upper bound: 0.0036697
time: 2.50 seconds

## BFS NS instance: NS_A1_B1_B2_B1_A2

### Backsubstitution after applying NS history:
0: 0.0057926, 0.0075508, 0.0058251, 0.0075060, -0.0017134, 0.0017257
1: -0.0010850, 0.0027074, -0.0007975, 0.0026204, -0.0037054, 0.0035048
2: -0.0049096, 0.0230447, -0.0042082, 0.0222547, -0.0271643, 0.0272530
3: -0.0045274, -0.0020741, -0.0044821, -0.0021367, -0.0023907, 0.0024080
4: -0.0006302, 0.0112732, -0.0004099, 0.0109692, -0.0115994, 0.0116831
5: -0.0022026, -0.0001088, -0.0021572, -0.0003256, -0.0018770, 0.0020484
6: 0.9902049, 0.9942927, 0.9907463, 0.9942095, -0.0040047, 0.0035465
7: -0.0146836, 0.0070235, -0.0141920, 0.0064733, -0.0211569, 0.0212155
8: -0.0059264, 0.0031888, -0.0044290, 0.0030164, -0.0089428, 0.0076177
9: -0.0136935, 0.0000932, -0.0133495, -0.0003381, -0.0133554, 0.0134427

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of NS_A1_B1_B2_B1_A2_B1

### Relational analysis result of NS_A1_B1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036594, upper bound: 0.0038219
time: 2.48 seconds

## Relational analysis of NS_A1_B1_B2_B1_A2_B2

### Relational analysis result of NS_A1_B1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036460, upper bound: 0.0037906
time: 2.35 seconds

## BFS NS instance: NS_A1_B1_B2_B2_A1

### Backsubstitution after applying NS history:
0: 0.0058060, 0.0074591, 0.0057908, 0.0075845, -0.0017785, 0.0016683
1: -0.0009667, 0.0025297, -0.0011013, 0.0027726, -0.0037394, 0.0036310
2: -0.0034762, 0.0227198, -0.0054356, 0.0230895, -0.0265657, 0.0281555
3: -0.0045088, -0.0022021, -0.0045300, -0.0020271, -0.0024817, 0.0023279
4: -0.0005396, 0.0106520, -0.0006426, 0.0115011, -0.0120407, 0.0112947
5: -0.0021099, -0.0001980, -0.0022366, -0.0000966, -0.0020133, 0.0020386
6: 0.9904276, 0.9941227, 0.9901741, 0.9943551, -0.0039276, 0.0039486
7: -0.0144814, 0.0058991, -0.0147114, 0.0074361, -0.0219175, 0.0206106
8: -0.0053106, 0.0028365, -0.0060113, 0.0033180, -0.0086286, 0.0088478
9: -0.0129904, -0.0000841, -0.0139515, 0.0001176, -0.0131081, 0.0138673

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 214

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of NS_A1_B1_B2_B2_A1_B1

### Relational analysis result of NS_A1_B1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036953, upper bound: 0.0036987
time: 2.33 seconds

## Relational analysis of NS_A1_B1_B2_B2_A1_B2

### Relational analysis result of NS_A1_B1_B2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0036460, upper bound: 0.0036690
time: 2.88 seconds

## BFS NS instance: NS_A1_B1_B2_B2_A2

### Backsubstitution after applying NS history:
0: 0.0057926, 0.0075508, 0.0057908, 0.0075845, -0.0017919, 0.0017601
1: -0.0010850, 0.0027074, -0.0011013, 0.0027726, -0.0038576, 0.0038087
2: -0.0049096, 0.0230447, -0.0054356, 0.0230895, -0.0279991, 0.0284803
3: -0.0045274, -0.0020741, -0.0045300, -0.0020271, -0.0025004, 0.0024559
4: -0.0006302, 0.0112732, -0.0006426, 0.0115011, -0.0121312, 0.0119158
5: -0.0022026, -0.0001088, -0.0022366, -0.0000966, -0.0021060, 0.0021278
6: 0.9902049, 0.9942927, 0.9901741, 0.9943551, -0.0041503, 0.0041186
7: -0.0146836, 0.0070235, -0.0147114, 0.0074361, -0.0221197, 0.0217349
8: -0.0059264, 0.0031888, -0.0060113, 0.0033180, -0.0092444, 0.0092000
9: -0.0136935, 0.0000932, -0.0139515, 0.0001176, -0.0138111, 0.0140447

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of NS_A1_B1_B2_B2_A2_B1

### Relational analysis result of NS_A1_B1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036953, upper bound: 0.0038120
time: 2.65 seconds

## Relational analysis of NS_A1_B1_B2_B2_A2_B2

### Relational analysis result of NS_A1_B1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036460, upper bound: 0.0037885
time: 3.01 seconds

## BFS NS instance: NS_A1_B2_A1_B1_B1

### Backsubstitution after applying NS history:
0: 0.0058038, 0.0074683, 0.0058140, 0.0074820, -0.0016783, 0.0016543
1: -0.0009862, 0.0025475, -0.0008958, 0.0025741, -0.0035603, 0.0034433
2: -0.0036196, 0.0227733, -0.0038345, 0.0225249, -0.0261445, 0.0266078
3: -0.0045118, -0.0021893, -0.0044976, -0.0021701, -0.0023418, 0.0023083
4: -0.0005545, 0.0107142, -0.0004853, 0.0108073, -0.0113618, 0.0111994
5: -0.0021191, -0.0001833, -0.0021330, -0.0002515, -0.0018677, 0.0019497
6: 0.9903909, 0.9941397, 0.9905611, 0.9941652, -0.0037743, 0.0035786
7: -0.0145147, 0.0060116, -0.0143602, 0.0061802, -0.0206949, 0.0203718
8: -0.0054119, 0.0028717, -0.0049411, 0.0029245, -0.0083365, 0.0078129
9: -0.0130608, -0.0000550, -0.0131661, -0.0001905, -0.0128702, 0.0131112

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of NS_A1_B2_A1_B1_B1_B1

### Relational analysis result of NS_A1_B2_A1_B1_B1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0033857, upper bound: 0.0034287
time: 2.45 seconds

## Relational analysis of NS_A1_B2_A1_B1_B1_B2

### Relational analysis result of NS_A1_B2_A1_B1_B1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0033784, upper bound: 0.0032963
time: 2.10 seconds

## BFS NS instance: NS_A1_B2_A1_B1_B2

### Backsubstitution after applying NS history:
0: 0.0058118, 0.0074558, 0.0057535, 0.0074875, -0.0016757, 0.0017023
1: -0.0009157, 0.0025233, -0.0014306, 0.0025846, -0.0035003, 0.0039539
2: -0.0034245, 0.0225796, -0.0039195, 0.0239943, -0.0274188, 0.0264991
3: -0.0045007, -0.0022067, -0.0045820, -0.0021625, -0.0023382, 0.0023753
4: -0.0005005, 0.0106296, -0.0008949, 0.0108441, -0.0113446, 0.0115245
5: -0.0021065, -0.0002365, -0.0021385, 0.0001517, -0.0022582, 0.0019021
6: 0.9905236, 0.9941166, 0.9895541, 0.9941753, -0.0036517, 0.0045625
7: -0.0143942, 0.0058586, -0.0152744, 0.0062469, -0.0206410, 0.0211329
8: -0.0050447, 0.0028238, -0.0077263, 0.0029454, -0.0079902, 0.0105501
9: -0.0129650, -0.0001607, -0.0132078, 0.0006116, -0.0135766, 0.0130471

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of NS_A1_B2_A1_B1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0035433, upper bound: 0.0032838
time: 5.96 seconds

## Relational analysis of NS_A1_B2_A1_B1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0033766, upper bound: 0.0032554
time: 2.36 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B1

### Backsubstitution after applying NS history:
0: 0.0058038, 0.0074683, 0.0057713, 0.0075697, -0.0017659, 0.0016970
1: -0.0009862, 0.0025475, -0.0012733, 0.0027439, -0.0037301, 0.0038207
2: -0.0036196, 0.0227733, -0.0052039, 0.0235620, -0.0271817, 0.0279772
3: -0.0045118, -0.0021893, -0.0045572, -0.0020478, -0.0024641, 0.0023679
4: -0.0005545, 0.0107142, -0.0007744, 0.0114007, -0.0119552, 0.0114885
5: -0.0021191, -0.0001833, -0.0022216, 0.0000331, -0.0021522, 0.0020383
6: 0.9903909, 0.9941397, 0.9898503, 0.9943277, -0.0039368, 0.0042894
7: -0.0145147, 0.0060116, -0.0150054, 0.0072543, -0.0217690, 0.0210171
8: -0.0054119, 0.0028717, -0.0069069, 0.0032611, -0.0086730, 0.0097787
9: -0.0130608, -0.0000550, -0.0138378, 0.0003756, -0.0134363, 0.0137829

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of NS_A1_B2_A1_B2_B1_B1

### Relational analysis result of NS_A1_B2_A1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0034550, upper bound: 0.0034255
time: 2.21 seconds

## Relational analysis of NS_A1_B2_A1_B2_B1_B2

### Relational analysis result of NS_A1_B2_A1_B2_B1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0034506, upper bound: 0.0032963
time: 2.57 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B2

### Backsubstitution after applying NS history:
0: 0.0058118, 0.0074558, 0.0057077, 0.0075763, -0.0017645, 0.0017481
1: -0.0009157, 0.0025233, -0.0018355, 0.0027566, -0.0036723, 0.0043588
2: -0.0034245, 0.0225796, -0.0053066, 0.0251068, -0.0285313, 0.0278862
3: -0.0045007, -0.0022067, -0.0046459, -0.0020386, -0.0024621, 0.0024392
4: -0.0005005, 0.0106296, -0.0012050, 0.0114452, -0.0119457, 0.0118346
5: -0.0021065, -0.0002365, -0.0022283, 0.0004570, -0.0025635, 0.0019918
6: 0.9905236, 0.9941166, 0.9887916, 0.9943398, -0.0038162, 0.0053250
7: -0.0143942, 0.0058586, -0.0159665, 0.0073349, -0.0217291, 0.0218251
8: -0.0050447, 0.0028238, -0.0098349, 0.0032863, -0.0083311, 0.0126587
9: -0.0129650, -0.0001607, -0.0138882, 0.0012188, -0.0141839, 0.0137275

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of NS_A1_B2_A1_B2_B2_B1

### Relational analysis result of NS_A1_B2_A1_B2_B2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0034624, upper bound: 0.0033840
time: 2.48 seconds

## Relational analysis of NS_A1_B2_A1_B2_B2_B2

### Relational analysis result of NS_A1_B2_A1_B2_B2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0034590, upper bound: 0.0032554
time: 2.31 seconds

## BFS NS instance: NS_A1_B2_A2_B1_B1

### Backsubstitution after applying NS history:
0: 0.0057888, 0.0075625, 0.0058095, 0.0074490, -0.0016601, 0.0017530
1: -0.0011183, 0.0027299, -0.0009356, 0.0025101, -0.0036284, 0.0036655
2: -0.0050914, 0.0231362, -0.0033183, 0.0226341, -0.0277255, 0.0264545
3: -0.0045327, -0.0020578, -0.0045039, -0.0022162, -0.0023165, 0.0024460
4: -0.0006557, 0.0113519, -0.0005157, 0.0105836, -0.0112392, 0.0118676
5: -0.0022143, -0.0000837, -0.0020996, -0.0002215, -0.0019928, 0.0020159
6: 0.9901422, 0.9943143, 0.9904863, 0.9941039, -0.0039617, 0.0038280
7: -0.0147405, 0.0071661, -0.0144281, 0.0057752, -0.0205157, 0.0215942
8: -0.0060998, 0.0032334, -0.0051481, 0.0027977, -0.0088975, 0.0083815
9: -0.0137826, 0.0001431, -0.0129129, -0.0001309, -0.0136517, 0.0130561

Time for backsubstitution: 2.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A1_B2_A2_B1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037068, upper bound: 0.0037614
time: 2.63 seconds

## Relational analysis of NS_A1_B2_A2_B1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036460, upper bound: 0.0037690
time: 2.44 seconds

## BFS NS instance: NS_A1_B2_A2_B1_B2

### Backsubstitution after applying NS history:
0: 0.0057888, 0.0075625, 0.0057962, 0.0075394, -0.0017506, 0.0017663
1: -0.0011183, 0.0027299, -0.0010534, 0.0026853, -0.0038036, 0.0037833
2: -0.0050914, 0.0231362, -0.0047316, 0.0229580, -0.0280494, 0.0278678
3: -0.0045327, -0.0020578, -0.0045225, -0.0020900, -0.0024427, 0.0024646
4: -0.0006557, 0.0113519, -0.0006060, 0.0111960, -0.0118517, 0.0119579
5: -0.0022143, -0.0000837, -0.0021911, -0.0001326, -0.0020817, 0.0021073
6: 0.9901422, 0.9943143, 0.9902643, 0.9942716, -0.0041295, 0.0040500
7: -0.0147405, 0.0071661, -0.0146296, 0.0068839, -0.0216244, 0.0217957
8: -0.0060998, 0.0032334, -0.0057620, 0.0031450, -0.0092448, 0.0089954
9: -0.0137826, 0.0001431, -0.0136062, 0.0000459, -0.0138285, 0.0137493

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A1_B2_A2_B1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037068, upper bound: 0.0037817
time: 2.67 seconds

## Relational analysis of NS_A1_B2_A2_B1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036697, upper bound: 0.0037906
time: 3.00 seconds

## BFS NS instance: NS_A1_B2_A2_B2_B1

### Backsubstitution after applying NS history:
0: 0.0057888, 0.0075625, 0.0057645, 0.0075392, -0.0017504, 0.0017980
1: -0.0011183, 0.0027299, -0.0013337, 0.0026849, -0.0038032, 0.0040636
2: -0.0050914, 0.0231362, -0.0047281, 0.0237279, -0.0288193, 0.0278643
3: -0.0045327, -0.0020578, -0.0045667, -0.0020903, -0.0024424, 0.0025089
4: -0.0006557, 0.0113519, -0.0008206, 0.0111945, -0.0118502, 0.0121725
5: -0.0022143, -0.0000837, -0.0021908, 0.0000786, -0.0022930, 0.0021071
6: 0.9901422, 0.9943143, 0.9897366, 0.9942713, -0.0041291, 0.0045778
7: -0.0147405, 0.0071661, -0.0151087, 0.0068811, -0.0216216, 0.0222747
8: -0.0060998, 0.0032334, -0.0072214, 0.0031442, -0.0092440, 0.0104548
9: -0.0137826, 0.0001431, -0.0136044, 0.0004662, -0.0142488, 0.0137476

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A1_B2_A2_B2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037971, upper bound: 0.0037612
time: 3.12 seconds

## Relational analysis of NS_A1_B2_A2_B2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037501, upper bound: 0.0037679
time: 2.97 seconds

## BFS NS instance: NS_A1_B2_A2_B2_B2

### Backsubstitution after applying NS history:
0: 0.0057888, 0.0075625, 0.0057530, 0.0076257, -0.0018369, 0.0018095
1: -0.0011183, 0.0027299, -0.0014353, 0.0028524, -0.0039707, 0.0041652
2: -0.0050914, 0.0231362, -0.0060796, 0.0240072, -0.0290986, 0.0292158
3: -0.0045327, -0.0020578, -0.0045827, -0.0019696, -0.0025631, 0.0025249
4: -0.0006557, 0.0113519, -0.0008985, 0.0117802, -0.0124358, 0.0122504
5: -0.0022143, -0.0000837, -0.0022783, 0.0001553, -0.0023696, 0.0021945
6: 0.9901422, 0.9943143, 0.9895453, 0.9944316, -0.0042894, 0.0047690
7: -0.0147405, 0.0071661, -0.0152824, 0.0079413, -0.0226818, 0.0224485
8: -0.0060998, 0.0032334, -0.0077507, 0.0034763, -0.0095761, 0.0109842
9: -0.0137826, 0.0001431, -0.0142673, 0.0006186, -0.0144012, 0.0144105

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of NS_A1_B2_A2_B2_B2_B1

### Relational analysis result of NS_A1_B2_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037444, upper bound: 0.0038120
time: 2.21 seconds

## Relational analysis of NS_A1_B2_A2_B2_B2_B2

### Relational analysis result of NS_A1_B2_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037501, upper bound: 0.0037885
time: 3.18 seconds

## BFS NS instance: NS_A2_B1_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.0057620, 0.0075416, 0.0058432, 0.0074482, -0.0016862, 0.0016984
1: -0.0013554, 0.0026895, -0.0006375, 0.0025085, -0.0038639, 0.0033270
2: -0.0047656, 0.0237875, -0.0033054, 0.0218152, -0.0265808, 0.0270929
3: -0.0045701, -0.0020869, -0.0044568, -0.0022174, -0.0023528, 0.0023699
4: -0.0008372, 0.0112107, -0.0002874, 0.0105780, -0.0114152, 0.0114982
5: -0.0021933, 0.0000950, -0.0020988, -0.0004462, -0.0017471, 0.0021938
6: 0.9896958, 0.9942757, 0.9910475, 0.9941025, -0.0044067, 0.0032282
7: -0.0151457, 0.0069105, -0.0139186, 0.0057652, -0.0209109, 0.0208291
8: -0.0073344, 0.0031534, -0.0035959, 0.0027945, -0.0101289, 0.0067493
9: -0.0136228, 0.0004987, -0.0129066, -0.0005780, -0.0130449, 0.0134053

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of NS_A2_B1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0035266, upper bound: 0.0034342
time: 2.22 seconds

## Relational analysis of NS_A2_B1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0033688, upper bound: 0.0034245
time: 2.30 seconds

## BFS NS instance: NS_A2_B1_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.0057698, 0.0075295, 0.0057825, 0.0074504, -0.0016806, 0.0017470
1: -0.0012864, 0.0026661, -0.0011738, 0.0025128, -0.0037993, 0.0038399
2: -0.0045767, 0.0235982, -0.0033402, 0.0232888, -0.0278655, 0.0269385
3: -0.0045592, -0.0021038, -0.0045415, -0.0022142, -0.0023450, 0.0024377
4: -0.0007844, 0.0111289, -0.0006982, 0.0105931, -0.0113775, 0.0118271
5: -0.0021810, 0.0000430, -0.0021011, -0.0000419, -0.0021392, 0.0021441
6: 0.9898255, 0.9942533, 0.9900376, 0.9941066, -0.0042811, 0.0042157
7: -0.0150280, 0.0067624, -0.0148354, 0.0057925, -0.0208204, 0.0215978
8: -0.0069755, 0.0031069, -0.0063890, 0.0028031, -0.0097786, 0.0094960
9: -0.0135302, 0.0003953, -0.0129237, 0.0002264, -0.0137566, 0.0133191

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of NS_A2_B1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0035320, upper bound: 0.0033809
time: 1.93 seconds

## Relational analysis of NS_A2_B1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0033644, upper bound: 0.0033631
time: 2.47 seconds

## BFS NS instance: NS_A2_B1_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.0057491, 0.0076331, 0.0058420, 0.0074122, -0.0016631, 0.0017912
1: -0.0014699, 0.0028668, -0.0006486, 0.0024388, -0.0039087, 0.0035154
2: -0.0061952, 0.0241023, -0.0027430, 0.0218458, -0.0280410, 0.0268453
3: -0.0045882, -0.0019592, -0.0044586, -0.0022676, -0.0023206, 0.0024993
4: -0.0009250, 0.0118303, -0.0002960, 0.0103343, -0.0112592, 0.0121262
5: -0.0022857, 0.0001814, -0.0020624, -0.0004378, -0.0018479, 0.0022438
6: 0.9894801, 0.9944453, 0.9910264, 0.9940357, -0.0045556, 0.0034189
7: -0.0153416, 0.0080320, -0.0139376, 0.0053240, -0.0206656, 0.0219696
8: -0.0079309, 0.0035047, -0.0036539, 0.0026563, -0.0105872, 0.0071586
9: -0.0143240, 0.0006705, -0.0126308, -0.0005612, -0.0137628, 0.0133013

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 214

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A2_B1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036721, upper bound: 0.0038232
time: 2.66 seconds

## Relational analysis of NS_A2_B1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036460, upper bound: 0.0038502
time: 2.66 seconds

## BFS NS instance: NS_A2_B1_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.0057491, 0.0076331, 0.0058251, 0.0075060, -0.0017569, 0.0018080
1: -0.0014699, 0.0028668, -0.0007975, 0.0026204, -0.0040903, 0.0036642
2: -0.0061952, 0.0241023, -0.0042082, 0.0222547, -0.0284499, 0.0283105
3: -0.0045882, -0.0019592, -0.0044821, -0.0021367, -0.0024515, 0.0025228
4: -0.0009250, 0.0118303, -0.0004099, 0.0109692, -0.0118942, 0.0122402
5: -0.0022857, 0.0001814, -0.0021572, -0.0003256, -0.0019601, 0.0023386
6: 0.9894801, 0.9944453, 0.9907463, 0.9942095, -0.0047294, 0.0036990
7: -0.0153416, 0.0080320, -0.0141920, 0.0064733, -0.0218149, 0.0222240
8: -0.0079309, 0.0035047, -0.0044290, 0.0030164, -0.0109473, 0.0079336
9: -0.0143240, 0.0006705, -0.0133495, -0.0003381, -0.0139860, 0.0140200

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A2_B1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036721, upper bound: 0.0038429
time: 2.23 seconds

## Relational analysis of NS_A2_B1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036460, upper bound: 0.0038747
time: 3.11 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A1

### Backsubstitution after applying NS history:
0: 0.0057674, 0.0075761, 0.0058069, 0.0074916, -0.0017241, 0.0017692
1: -0.0013075, 0.0027563, -0.0009587, 0.0025926, -0.0039001, 0.0037151
2: -0.0053042, 0.0236561, -0.0039838, 0.0226979, -0.0280021, 0.0276399
3: -0.0045626, -0.0020388, -0.0045075, -0.0021568, -0.0024058, 0.0024687
4: -0.0008006, 0.0114441, -0.0005335, 0.0108719, -0.0116725, 0.0119776
5: -0.0022281, 0.0000589, -0.0021427, -0.0002040, -0.0020241, 0.0022016
6: 0.9897859, 0.9943395, 0.9904426, 0.9941829, -0.0043970, 0.0038970
7: -0.0150640, 0.0073330, -0.0144678, 0.0062973, -0.0213612, 0.0218008
8: -0.0070853, 0.0032857, -0.0052689, 0.0029612, -0.0100465, 0.0085547
9: -0.0138870, 0.0004269, -0.0132394, -0.0000961, -0.0137909, 0.0136663

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of NS_A2_B1_B2_B1_A1_A1

### Relational analysis result of NS_A2_B1_B2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0034355, upper bound: 0.0034797
time: 2.94 seconds

## Relational analysis of NS_A2_B1_B2_B1_A1_A2

### Relational analysis result of NS_A2_B1_B2_B1_A1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0033123, upper bound: 0.0034779
time: 1.85 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A2

### Backsubstitution after applying NS history:
0: 0.0057038, 0.0075826, 0.0058142, 0.0074793, -0.0017755, 0.0017684
1: -0.0018702, 0.0027690, -0.0008941, 0.0025688, -0.0044390, 0.0036631
2: -0.0054062, 0.0252021, -0.0037919, 0.0225204, -0.0279266, 0.0289940
3: -0.0046514, -0.0020297, -0.0044973, -0.0021739, -0.0024775, 0.0024676
4: -0.0012315, 0.0114884, -0.0004840, 0.0107888, -0.0120203, 0.0119723
5: -0.0022347, 0.0004831, -0.0021303, -0.0002527, -0.0019820, 0.0026134
6: 0.9887263, 0.9943516, 0.9905642, 0.9941601, -0.0054339, 0.0037874
7: -0.0160258, 0.0074130, -0.0143573, 0.0061467, -0.0221726, 0.0217704
8: -0.0100155, 0.0033108, -0.0049325, 0.0029141, -0.0129296, 0.0082433
9: -0.0139370, 0.0012708, -0.0131452, -0.0001930, -0.0137440, 0.0144161

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of NS_A2_B1_B2_B1_A2_A1

### Relational analysis result of NS_A2_B1_B2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0034084, upper bound: 0.0034899
time: 2.40 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2_A2

### Relational analysis result of NS_A2_B1_B2_B1_A2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0032860, upper bound: 0.0034889
time: 6.08 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A1

### Backsubstitution after applying NS history:
0: 0.0057606, 0.0075450, 0.0057908, 0.0075845, -0.0018239, 0.0017543
1: -0.0013678, 0.0026961, -0.0011013, 0.0027726, -0.0041404, 0.0037974
2: -0.0048187, 0.0238218, -0.0054356, 0.0230895, -0.0279081, 0.0292575
3: -0.0045721, -0.0020822, -0.0045300, -0.0020271, -0.0025450, 0.0024478
4: -0.0008468, 0.0112337, -0.0006426, 0.0115011, -0.0123479, 0.0118764
5: -0.0021967, 0.0001044, -0.0022366, -0.0000966, -0.0021001, 0.0023410
6: 0.9896723, 0.9942820, 0.9901741, 0.9943551, -0.0046828, 0.0041079
7: -0.0151671, 0.0069522, -0.0147114, 0.0074361, -0.0226032, 0.0216636
8: -0.0073994, 0.0031664, -0.0060113, 0.0033180, -0.0107174, 0.0091777
9: -0.0136489, 0.0005174, -0.0139515, 0.0001176, -0.0137665, 0.0144689

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of NS_A2_B1_B2_B2_A1_B1

### Relational analysis result of NS_A2_B1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036608, upper bound: 0.0037971
time: 2.76 seconds

## Relational analysis of NS_A2_B1_B2_B2_A1_B2

### Relational analysis result of NS_A2_B1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036486, upper bound: 0.0037501
time: 2.61 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A2

### Backsubstitution after applying NS history:
0: 0.0057491, 0.0076331, 0.0057908, 0.0075845, -0.0018354, 0.0018424
1: -0.0014699, 0.0028668, -0.0011013, 0.0027726, -0.0042425, 0.0039681
2: -0.0061952, 0.0241023, -0.0054356, 0.0230895, -0.0292847, 0.0295379
3: -0.0045882, -0.0019592, -0.0045300, -0.0020271, -0.0025611, 0.0025708
4: -0.0009250, 0.0118303, -0.0006426, 0.0115011, -0.0124260, 0.0124729
5: -0.0022857, 0.0001814, -0.0022366, -0.0000966, -0.0021892, 0.0024180
6: 0.9894801, 0.9944453, 0.9901741, 0.9943551, -0.0048750, 0.0042711
7: -0.0153416, 0.0080320, -0.0147114, 0.0074361, -0.0227777, 0.0227434
8: -0.0079309, 0.0035047, -0.0060113, 0.0033180, -0.0112489, 0.0095159
9: -0.0143240, 0.0006705, -0.0139515, 0.0001176, -0.0144417, 0.0146220

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of NS_A2_B1_B2_B2_A2_B1

### Relational analysis result of NS_A2_B1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036608, upper bound: 0.0039001
time: 2.39 seconds

## Relational analysis of NS_A2_B1_B2_B2_A2_B2

### Relational analysis result of NS_A2_B1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036486, upper bound: 0.0038747
time: 2.85 seconds

## BFS NS instance: NS_A2_B2_B1_A1_A1

### Backsubstitution after applying NS history:
0: 0.0057758, 0.0075155, 0.0057978, 0.0075226, -0.0017468, 0.0017177
1: -0.0012334, 0.0026389, -0.0010393, 0.0026526, -0.0038860, 0.0036781
2: -0.0043569, 0.0234524, -0.0044679, 0.0229191, -0.0272760, 0.0279203
3: -0.0045509, -0.0021234, -0.0045202, -0.0021135, -0.0024373, 0.0023968
4: -0.0007438, 0.0110337, -0.0005951, 0.0110818, -0.0118256, 0.0116288
5: -0.0021668, 0.0000030, -0.0021740, -0.0001433, -0.0020235, 0.0021770
6: 0.9899254, 0.9942272, 0.9902909, 0.9942404, -0.0043150, 0.0039364
7: -0.0149373, 0.0065900, -0.0146054, 0.0066770, -0.0216143, 0.0211954
8: -0.0066992, 0.0030529, -0.0056883, 0.0030802, -0.0097794, 0.0087412
9: -0.0134224, 0.0003158, -0.0134768, 0.0000246, -0.0134470, 0.0137926

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of NS_A2_B2_B1_A1_A1_A1

### Relational analysis result of NS_A2_B2_B1_A1_A1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0036009, upper bound: 0.0033867
time: 1.81 seconds

## Relational analysis of NS_A2_B2_B1_A1_A1_A2

### Relational analysis result of NS_A2_B2_B1_A1_A1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0034543, upper bound: 0.0033694
time: 2.43 seconds

## BFS NS instance: NS_A2_B2_B1_A1_A2

### Backsubstitution after applying NS history:
0: 0.0057157, 0.0075206, 0.0058052, 0.0075107, -0.0017950, 0.0017154
1: -0.0017650, 0.0026489, -0.0009734, 0.0026296, -0.0043946, 0.0036222
2: -0.0044375, 0.0249130, -0.0042823, 0.0227380, -0.0271755, 0.0291953
3: -0.0046348, -0.0021162, -0.0045098, -0.0021301, -0.0025047, 0.0023936
4: -0.0011509, 0.0110686, -0.0005447, 0.0110013, -0.0121523, 0.0116133
5: -0.0021720, 0.0004038, -0.0021620, -0.0001930, -0.0019790, 0.0025658
6: 0.9889246, 0.9942367, 0.9904150, 0.9942183, -0.0052938, 0.0038217
7: -0.0158460, 0.0066532, -0.0144927, 0.0065315, -0.0223774, 0.0211460
8: -0.0094675, 0.0030727, -0.0053450, 0.0030346, -0.0125021, 0.0084178
9: -0.0134619, 0.0011130, -0.0133858, -0.0000742, -0.0133877, 0.0144988

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of NS_A2_B2_B1_A1_A2_A1

### Relational analysis result of NS_A2_B2_B1_A1_A2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0035433, upper bound: 0.0033809
time: 2.73 seconds

## Relational analysis of NS_A2_B2_B1_A1_A2_A2

### Relational analysis result of NS_A2_B2_B1_A1_A2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0033767, upper bound: 0.0033631
time: 3.07 seconds

## BFS NS instance: NS_A2_B2_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.0057455, 0.0076467, 0.0058095, 0.0074490, -0.0017034, 0.0018371
1: -0.0015011, 0.0028930, -0.0009356, 0.0025101, -0.0040112, 0.0038285
2: -0.0064065, 0.0241880, -0.0033183, 0.0226341, -0.0290406, 0.0275062
3: -0.0045931, -0.0019404, -0.0045039, -0.0022162, -0.0023769, 0.0025635
4: -0.0009488, 0.0119218, -0.0005157, 0.0105836, -0.0115324, 0.0124375
5: -0.0022994, 0.0002049, -0.0020996, -0.0002215, -0.0020779, 0.0023045
6: 0.9894214, 0.9944704, 0.9904863, 0.9941039, -0.0046825, 0.0039842
7: -0.0153949, 0.0081977, -0.0144281, 0.0057752, -0.0211701, 0.0226258
8: -0.0080934, 0.0035566, -0.0051481, 0.0027977, -0.0108911, 0.0087047
9: -0.0144277, 0.0007173, -0.0129129, -0.0001309, -0.0142967, 0.0136302

Time for backsubstitution: 2.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 214

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A2_B2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036721, upper bound: 0.0038232
time: 3.27 seconds

## Relational analysis of NS_A2_B2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036690, upper bound: 0.0038501
time: 2.87 seconds

## BFS NS instance: NS_A2_B2_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.0057455, 0.0076467, 0.0057962, 0.0075394, -0.0017939, 0.0018505
1: -0.0015011, 0.0028930, -0.0010534, 0.0026853, -0.0041864, 0.0039464
2: -0.0064065, 0.0241880, -0.0047316, 0.0229580, -0.0293645, 0.0289196
3: -0.0045931, -0.0019404, -0.0045225, -0.0020900, -0.0025032, 0.0025821
4: -0.0009488, 0.0119218, -0.0006060, 0.0111960, -0.0121449, 0.0125278
5: -0.0022994, 0.0002049, -0.0021911, -0.0001326, -0.0021668, 0.0023959
6: 0.9894214, 0.9944704, 0.9902643, 0.9942716, -0.0048503, 0.0042061
7: -0.0153949, 0.0081977, -0.0146296, 0.0068839, -0.0222788, 0.0228273
8: -0.0080934, 0.0035566, -0.0057620, 0.0031450, -0.0112384, 0.0093186
9: -0.0144277, 0.0007173, -0.0136062, 0.0000459, -0.0144735, 0.0143234

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A2_B2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036987, upper bound: 0.0038430
time: 2.87 seconds

## Relational analysis of NS_A2_B2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036690, upper bound: 0.0038748
time: 2.45 seconds

## BFS NS instance: NS_A2_B2_B2_B1_A1

### Backsubstitution after applying NS history:
0: 0.0057639, 0.0075898, 0.0057659, 0.0075358, -0.0017720, 0.0018240
1: -0.0013389, 0.0027829, -0.0013212, 0.0026783, -0.0040172, 0.0041041
2: -0.0055187, 0.0237423, -0.0046751, 0.0236936, -0.0292123, 0.0284174
3: -0.0045675, -0.0020197, -0.0045647, -0.0020950, -0.0024725, 0.0025451
4: -0.0008246, 0.0115371, -0.0008110, 0.0111716, -0.0119962, 0.0123481
5: -0.0022420, 0.0000826, -0.0021874, 0.0000692, -0.0023112, 0.0022700
6: 0.9897267, 0.9943650, 0.9897602, 0.9942650, -0.0045383, 0.0046048
7: -0.0151176, 0.0075013, -0.0150873, 0.0068396, -0.0219572, 0.0225886
8: -0.0072486, 0.0033385, -0.0071563, 0.0031311, -0.0103797, 0.0104948
9: -0.0139922, 0.0004740, -0.0135785, 0.0004474, -0.0144396, 0.0140524

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 96

## Relational analysis of NS_A2_B2_B2_B1_A1_A1

### Relational analysis result of NS_A2_B2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036722, upper bound: 0.0037857
time: 2.09 seconds

## Relational analysis of NS_A2_B2_B2_B1_A1_A2

### Relational analysis result of NS_A2_B2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036722, upper bound: 0.0038232
time: 6.56 seconds

## BFS NS instance: NS_A2_B2_B2_B1_A2

### Backsubstitution after applying NS history:
0: 0.0057003, 0.0075965, 0.0057736, 0.0075237, -0.0018234, 0.0018229
1: -0.0019011, 0.0027958, -0.0012531, 0.0026548, -0.0045558, 0.0040489
2: -0.0056229, 0.0252869, -0.0044851, 0.0235065, -0.0291294, 0.0297720
3: -0.0046563, -0.0020104, -0.0045540, -0.0021120, -0.0025443, 0.0025436
4: -0.0012552, 0.0115822, -0.0007589, 0.0110892, -0.0123444, 0.0123411
5: -0.0022487, 0.0005064, -0.0021751, 0.0000179, -0.0022666, 0.0026815
6: 0.9886681, 0.9943774, 0.9898883, 0.9942424, -0.0055743, 0.0044891
7: -0.0160786, 0.0075830, -0.0149709, 0.0066905, -0.0227691, 0.0225539
8: -0.0101764, 0.0033640, -0.0068018, 0.0030844, -0.0132608, 0.0101658
9: -0.0140433, 0.0013172, -0.0134852, 0.0003453, -0.0143886, 0.0148024

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of NS_A2_B2_B2_B1_A2_A1

### Relational analysis result of NS_A2_B2_B2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0034268, upper bound: 0.0034899
time: 2.48 seconds

## Relational analysis of NS_A2_B2_B2_B1_A2_A2

### Relational analysis result of NS_A2_B2_B2_B1_A2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0033167, upper bound: 0.0034888
time: 2.19 seconds

## BFS NS instance: NS_A2_B2_B2_B2_A1

### Backsubstitution after applying NS history:
0: 0.0057572, 0.0075594, 0.0057530, 0.0076257, -0.0018685, 0.0018064
1: -0.0013975, 0.0027240, -0.0014353, 0.0028524, -0.0042500, 0.0041593
2: -0.0050434, 0.0239035, -0.0060796, 0.0240072, -0.0290506, 0.0299831
3: -0.0045768, -0.0020621, -0.0045827, -0.0019696, -0.0026072, 0.0025206
4: -0.0008695, 0.0113312, -0.0008985, 0.0117802, -0.0126497, 0.0122296
5: -0.0022112, 0.0001268, -0.0022783, 0.0001553, -0.0023665, 0.0024051
6: 0.9896164, 0.9943086, 0.9895453, 0.9944316, -0.0048152, 0.0047633
7: -0.0152179, 0.0071285, -0.0152824, 0.0079413, -0.0231592, 0.0224109
8: -0.0075541, 0.0032216, -0.0077507, 0.0034763, -0.0110304, 0.0109724
9: -0.0137591, 0.0005620, -0.0142673, 0.0006186, -0.0143777, 0.0148293

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of NS_A2_B2_B2_B2_A1_B1

### Relational analysis result of NS_A2_B2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036803, upper bound: 0.0037971
time: 2.64 seconds

## Relational analysis of NS_A2_B2_B2_B2_A1_B2

### Relational analysis result of NS_A2_B2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036734, upper bound: 0.0037501
time: 2.45 seconds

## BFS NS instance: NS_A2_B2_B2_B2_A2

### Backsubstitution after applying NS history:
0: 0.0057455, 0.0076467, 0.0057530, 0.0076257, -0.0018802, 0.0018937
1: -0.0015011, 0.0028930, -0.0014353, 0.0028524, -0.0043535, 0.0043283
2: -0.0064065, 0.0241880, -0.0060796, 0.0240072, -0.0304137, 0.0302676
3: -0.0045931, -0.0019404, -0.0045827, -0.0019696, -0.0026236, 0.0026424
4: -0.0009488, 0.0119218, -0.0008985, 0.0117802, -0.0127290, 0.0128203
5: -0.0022994, 0.0002049, -0.0022783, 0.0001553, -0.0024547, 0.0024831
6: 0.9894214, 0.9944704, 0.9895453, 0.9944316, -0.0050102, 0.0049251
7: -0.0153949, 0.0081977, -0.0152824, 0.0079413, -0.0233362, 0.0234801
8: -0.0080934, 0.0035566, -0.0077507, 0.0034763, -0.0115697, 0.0113073
9: -0.0144277, 0.0007173, -0.0142673, 0.0006186, -0.0150463, 0.0149846

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A2_B2_B2_B2_A2_A1

### Relational analysis result of NS_A2_B2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036994, upper bound: 0.0038429
time: 2.65 seconds

## Relational analysis of NS_A2_B2_B2_B2_A2_A2

### Relational analysis result of NS_A2_B2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036734, upper bound: 0.0038748
time: 2.72 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 7.32 seconds
NS_A1_B1_B1_B1_A1_A1, status: Status.VERIFIED, split count: 6, time: 7.32
Output dim: 6, lower bound: -0.0034128, upper bound: 0.0033857
NS_A1_B1_B1_B1_A1_A2, status: Status.VERIFIED, split count: 6, time: 7.32
Output dim: 6, lower bound: -0.0032687, upper bound: 0.0033784
NS_A1_B1_B1_B1_A2_A1, status: Status.VERIFIED, split count: 6, time: 7.32
Output dim: 6, lower bound: -0.0033714, upper bound: 0.0033827
NS_A1_B1_B1_B1_A2_A2, status: Status.VERIFIED, split count: 6, time: 7.32
Output dim: 6, lower bound: -0.0032362, upper bound: 0.0033766
NS_A1_B1_B1_B2_A1_A1, status: Status.VERIFIED, split count: 6, time: 7.32
Output dim: 6, lower bound: -0.0035001, upper bound: 0.0033863
NS_A1_B1_B1_B2_A1_A2, status: Status.VERIFIED, split count: 6, time: 7.32
Output dim: 6, lower bound: -0.0033459, upper bound: 0.0033791
NS_A1_B1_B1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 7.32
Output dim: 6, lower bound: -0.0032600, upper bound: 0.0035432
NS_A1_B1_B1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 7.32
Output dim: 6, lower bound: -0.0033036, upper bound: 0.0033767
NS_A1_B1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.32
Output dim: 6, lower bound: -0.0036594, upper bound: 0.0037068
NS_A1_B1_B2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 7.32
Output dim: 6, lower bound: -0.0036460, upper bound: 0.0036697
NS_A1_B1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.32
Output dim: 6, lower bound: -0.0036594, upper bound: 0.0038219
NS_A1_B1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.32
Output dim: 6, lower bound: -0.0036460, upper bound: 0.0037906
NS_A1_B1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.32
Output dim: 6, lower bound: -0.0036953, upper bound: 0.0036987
NS_A1_B1_B2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 7.32
Output dim: 6, lower bound: -0.0036460, upper bound: 0.0036690
NS_A1_B1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.32
Output dim: 6, lower bound: -0.0036953, upper bound: 0.0038120
NS_A1_B1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.32
Output dim: 6, lower bound: -0.0036460, upper bound: 0.0037885
NS_A1_B2_A1_B1_B1_B1, status: Status.VERIFIED, split count: 6, time: 7.32
Output dim: 6, lower bound: -0.0033857, upper bound: 0.0034287
NS_A1_B2_A1_B1_B1_B2, status: Status.VERIFIED, split count: 6, time: 7.32
Output dim: 6, lower bound: -0.0033784, upper bound: 0.0032963
NS_A1_B2_A1_B1_B2_A1, status: Status.VERIFIED, split count: 6, time: 7.32
Output dim: 6, lower bound: -0.0035433, upper bound: 0.0032838
NS_A1_B2_A1_B1_B2_A2, status: Status.VERIFIED, split count: 6, time: 7.32
Output dim: 6, lower bound: -0.0033766, upper bound: 0.0032554
NS_A1_B2_A1_B2_B1_B1, status: Status.VERIFIED, split count: 6, time: 7.32
Output dim: 6, lower bound: -0.0034550, upper bound: 0.0034255
NS_A1_B2_A1_B2_B1_B2, status: Status.VERIFIED, split count: 6, time: 7.32
Output dim: 6, lower bound: -0.0034506, upper bound: 0.0032963
NS_A1_B2_A1_B2_B2_B1, status: Status.VERIFIED, split count: 6, time: 7.32
Output dim: 6, lower bound: -0.0034624, upper bound: 0.0033840
NS_A1_B2_A1_B2_B2_B2, status: Status.VERIFIED, split count: 6, time: 7.32
Output dim: 6, lower bound: -0.0034590, upper bound: 0.0032554
NS_A1_B2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 7.32
Output dim: 6, lower bound: -0.0037068, upper bound: 0.0037614
NS_A1_B2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 7.32
Output dim: 6, lower bound: -0.0036460, upper bound: 0.0037690
NS_A1_B2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 7.32
Output dim: 6, lower bound: -0.0037068, upper bound: 0.0037817
NS_A1_B2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 7.32
Output dim: 6, lower bound: -0.0036697, upper bound: 0.0037906
NS_A1_B2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 7.32
Output dim: 6, lower bound: -0.0037971, upper bound: 0.0037612
NS_A1_B2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 7.32
Output dim: 6, lower bound: -0.0037501, upper bound: 0.0037679
NS_A1_B2_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 7.32
Output dim: 6, lower bound: -0.0037444, upper bound: 0.0038120
NS_A1_B2_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 7.32
Output dim: 6, lower bound: -0.0037501, upper bound: 0.0037885
NS_A2_B1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 6, time: 7.32
Output dim: 6, lower bound: -0.0035266, upper bound: 0.0034342
NS_A2_B1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 6, time: 7.32
Output dim: 6, lower bound: -0.0033688, upper bound: 0.0034245
NS_A2_B1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 6, time: 7.32
Output dim: 6, lower bound: -0.0035320, upper bound: 0.0033809
NS_A2_B1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 6, time: 7.32
Output dim: 6, lower bound: -0.0033644, upper bound: 0.0033631
NS_A2_B1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 7.32
Output dim: 6, lower bound: -0.0036721, upper bound: 0.0038232
NS_A2_B1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 7.32
Output dim: 6, lower bound: -0.0036460, upper bound: 0.0038502
NS_A2_B1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 7.32
Output dim: 6, lower bound: -0.0036721, upper bound: 0.0038429
NS_A2_B1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 7.32
Output dim: 6, lower bound: -0.0036460, upper bound: 0.0038747
NS_A2_B1_B2_B1_A1_A1, status: Status.VERIFIED, split count: 6, time: 7.32
Output dim: 6, lower bound: -0.0034355, upper bound: 0.0034797
NS_A2_B1_B2_B1_A1_A2, status: Status.VERIFIED, split count: 6, time: 7.32
Output dim: 6, lower bound: -0.0033123, upper bound: 0.0034779
NS_A2_B1_B2_B1_A2_A1, status: Status.VERIFIED, split count: 6, time: 7.32
Output dim: 6, lower bound: -0.0034084, upper bound: 0.0034899
NS_A2_B1_B2_B1_A2_A2, status: Status.VERIFIED, split count: 6, time: 7.32
Output dim: 6, lower bound: -0.0032860, upper bound: 0.0034889
NS_A2_B1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.32
Output dim: 6, lower bound: -0.0036608, upper bound: 0.0037971
NS_A2_B1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.32
Output dim: 6, lower bound: -0.0036486, upper bound: 0.0037501
NS_A2_B1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.32
Output dim: 6, lower bound: -0.0036608, upper bound: 0.0039001
NS_A2_B1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.32
Output dim: 6, lower bound: -0.0036486, upper bound: 0.0038747
NS_A2_B2_B1_A1_A1_A1, status: Status.VERIFIED, split count: 6, time: 7.32
Output dim: 6, lower bound: -0.0036009, upper bound: 0.0033867
NS_A2_B2_B1_A1_A1_A2, status: Status.VERIFIED, split count: 6, time: 7.32
Output dim: 6, lower bound: -0.0034543, upper bound: 0.0033694
NS_A2_B2_B1_A1_A2_A1, status: Status.VERIFIED, split count: 6, time: 7.32
Output dim: 6, lower bound: -0.0035433, upper bound: 0.0033809
NS_A2_B2_B1_A1_A2_A2, status: Status.VERIFIED, split count: 6, time: 7.32
Output dim: 6, lower bound: -0.0033767, upper bound: 0.0033631
NS_A2_B2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 7.32
Output dim: 6, lower bound: -0.0036721, upper bound: 0.0038232
NS_A2_B2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 7.32
Output dim: 6, lower bound: -0.0036690, upper bound: 0.0038501
NS_A2_B2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 7.32
Output dim: 6, lower bound: -0.0036987, upper bound: 0.0038430
NS_A2_B2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 7.32
Output dim: 6, lower bound: -0.0036690, upper bound: 0.0038748
NS_A2_B2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 7.32
Output dim: 6, lower bound: -0.0036722, upper bound: 0.0037857
NS_A2_B2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 7.32
Output dim: 6, lower bound: -0.0036722, upper bound: 0.0038232
NS_A2_B2_B2_B1_A2_A1, status: Status.VERIFIED, split count: 6, time: 7.32
Output dim: 6, lower bound: -0.0034268, upper bound: 0.0034899
NS_A2_B2_B2_B1_A2_A2, status: Status.VERIFIED, split count: 6, time: 7.32
Output dim: 6, lower bound: -0.0033167, upper bound: 0.0034888
NS_A2_B2_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.32
Output dim: 6, lower bound: -0.0036803, upper bound: 0.0037971
NS_A2_B2_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.32
Output dim: 6, lower bound: -0.0036734, upper bound: 0.0037501
NS_A2_B2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 7.32
Output dim: 6, lower bound: -0.0036994, upper bound: 0.0038429
NS_A2_B2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 7.32
Output dim: 6, lower bound: -0.0036734, upper bound: 0.0038748

## BFS NS instance: NS_A1_B1_B2_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.0058074, 0.0074558, 0.0058430, 0.0074619, -0.0016545, 0.0016128
1: -0.0009543, 0.0025234, -0.0006394, 0.0025352, -0.0034895, 0.0031628
2: -0.0034252, 0.0226858, -0.0035204, 0.0218205, -0.0252457, 0.0262062
3: -0.0045068, -0.0022067, -0.0044571, -0.0021981, -0.0023087, 0.0022505
4: -0.0005301, 0.0106299, -0.0002889, 0.0106712, -0.0112013, 0.0109188
5: -0.0021066, -0.0002073, -0.0021127, -0.0004448, -0.0016618, 0.0019054
6: 0.9904508, 0.9941167, 0.9910439, 0.9941280, -0.0036772, 0.0030727
7: -0.0144602, 0.0058591, -0.0139219, 0.0059338, -0.0203940, 0.0197810
8: -0.0052460, 0.0028240, -0.0036059, 0.0028474, -0.0080934, 0.0064299
9: -0.0129654, -0.0001027, -0.0130121, -0.0005751, -0.0123903, 0.0129093

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of NS_A1_B1_B2_B1_A1_B1_B1

### Relational analysis result of NS_A1_B1_B2_B1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0033733, upper bound: 0.0034287
time: 2.40 seconds

## Relational analysis of NS_A1_B1_B2_B1_A1_B1_B2

### Relational analysis result of NS_A1_B1_B2_B1_A1_B1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0033669, upper bound: 0.0032963
time: 2.51 seconds

## BFS NS instance: NS_A1_B1_B2_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.0057939, 0.0075474, 0.0058430, 0.0074619, -0.0016680, 0.0017044
1: -0.0010732, 0.0027008, -0.0006394, 0.0025352, -0.0036083, 0.0033402
2: -0.0048561, 0.0230122, -0.0035204, 0.0218205, -0.0266766, 0.0265326
3: -0.0045256, -0.0020788, -0.0044571, -0.0021981, -0.0023274, 0.0023783
4: -0.0006211, 0.0112500, -0.0002889, 0.0106712, -0.0112923, 0.0115389
5: -0.0021991, -0.0001178, -0.0021127, -0.0004448, -0.0017544, 0.0019950
6: 0.9902272, 0.9942865, 0.9910439, 0.9941280, -0.0039008, 0.0032426
7: -0.0146634, 0.0069816, -0.0139219, 0.0059338, -0.0205972, 0.0209034
8: -0.0058648, 0.0031756, -0.0036059, 0.0028474, -0.0087122, 0.0067816
9: -0.0136672, 0.0000755, -0.0130121, -0.0005751, -0.0130922, 0.0130875

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of NS_A1_B1_B2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_B2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0035171, upper bound: 0.0035264
time: 2.62 seconds

## Relational analysis of NS_A1_B1_B2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_B2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0034322, upper bound: 0.0035284
time: 2.99 seconds

## BFS NS instance: NS_A1_B1_B2_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.0058014, 0.0075357, 0.0057817, 0.0074675, -0.0016661, 0.0017540
1: -0.0010071, 0.0026780, -0.0011815, 0.0025460, -0.0035531, 0.0038595
2: -0.0046725, 0.0228307, -0.0036077, 0.0233100, -0.0279825, 0.0264384
3: -0.0045151, -0.0020952, -0.0045427, -0.0021904, -0.0023248, 0.0024474
4: -0.0005705, 0.0111704, -0.0007041, 0.0107090, -0.0112795, 0.0118745
5: -0.0021872, -0.0001676, -0.0021184, -0.0000361, -0.0021512, 0.0019508
6: 0.9903516, 0.9942647, 0.9900231, 0.9941383, -0.0037867, 0.0042416
7: -0.0145504, 0.0068375, -0.0148486, 0.0060022, -0.0205527, 0.0216862
8: -0.0055208, 0.0031305, -0.0064293, 0.0028688, -0.0083896, 0.0095598
9: -0.0135772, -0.0000236, -0.0130549, 0.0002380, -0.0138152, 0.0130313

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of NS_A1_B1_B2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_B2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0035159, upper bound: 0.0035084
time: 2.15 seconds

## Relational analysis of NS_A1_B1_B2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_B2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0034255, upper bound: 0.0035089
time: 2.58 seconds

## BFS NS instance: NS_A1_B1_B2_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.0058074, 0.0074558, 0.0058091, 0.0075392, -0.0017318, 0.0016467
1: -0.0009543, 0.0025234, -0.0009389, 0.0026848, -0.0036392, 0.0034622
2: -0.0034252, 0.0226858, -0.0047277, 0.0226432, -0.0260685, 0.0274135
3: -0.0045068, -0.0022067, -0.0045044, -0.0020903, -0.0024165, 0.0022977
4: -0.0005301, 0.0106299, -0.0005183, 0.0111944, -0.0117245, 0.0111482
5: -0.0021066, -0.0002073, -0.0021908, -0.0002190, -0.0018876, 0.0019835
6: 0.9904508, 0.9941167, 0.9904799, 0.9942712, -0.0038204, 0.0036367
7: -0.0144602, 0.0058591, -0.0144338, 0.0068808, -0.0213411, 0.0202929
8: -0.0052460, 0.0028240, -0.0051655, 0.0031441, -0.0083901, 0.0079894
9: -0.0129654, -0.0001027, -0.0136043, -0.0001259, -0.0128394, 0.0135015

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of NS_A1_B1_B2_B2_A1_B1_B1

### Relational analysis result of NS_A1_B1_B2_B2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0034134, upper bound: 0.0034255
time: 2.25 seconds

## Relational analysis of NS_A1_B1_B2_B2_A1_B1_B2

### Relational analysis result of NS_A1_B1_B2_B2_A1_B1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0034101, upper bound: 0.0032963
time: 2.57 seconds

## BFS NS instance: NS_A1_B1_B2_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.0057939, 0.0075474, 0.0058091, 0.0075392, -0.0017453, 0.0017383
1: -0.0010732, 0.0027008, -0.0009389, 0.0026848, -0.0037580, 0.0036396
2: -0.0048561, 0.0230122, -0.0047277, 0.0226432, -0.0274994, 0.0277400
3: -0.0045256, -0.0020788, -0.0045044, -0.0020903, -0.0024353, 0.0024255
4: -0.0006211, 0.0112500, -0.0005183, 0.0111944, -0.0118155, 0.0117682
5: -0.0021991, -0.0001178, -0.0021908, -0.0002190, -0.0019801, 0.0020731
6: 0.9902272, 0.9942865, 0.9904799, 0.9942712, -0.0040439, 0.0038065
7: -0.0146634, 0.0069816, -0.0144338, 0.0068808, -0.0215442, 0.0214153
8: -0.0058648, 0.0031756, -0.0051655, 0.0031441, -0.0090089, 0.0083411
9: -0.0136672, 0.0000755, -0.0136043, -0.0001259, -0.0135413, 0.0136797

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of NS_A1_B1_B2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_B2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0035171, upper bound: 0.0035251
time: 2.78 seconds

## Relational analysis of NS_A1_B1_B2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_B2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0034773, upper bound: 0.0035275
time: 3.13 seconds

## BFS NS instance: NS_A1_B1_B2_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.0058014, 0.0075357, 0.0057443, 0.0075469, -0.0017455, 0.0017914
1: -0.0010071, 0.0026780, -0.0015122, 0.0026998, -0.0037069, 0.0041902
2: -0.0046725, 0.0228307, -0.0048485, 0.0242186, -0.0288911, 0.0276792
3: -0.0045151, -0.0020952, -0.0045949, -0.0020795, -0.0024356, 0.0024996
4: -0.0005705, 0.0111704, -0.0009574, 0.0112467, -0.0118172, 0.0121278
5: -0.0021872, -0.0001676, -0.0021986, 0.0002133, -0.0024005, 0.0020311
6: 0.9903516, 0.9942647, 0.9894004, 0.9942855, -0.0039339, 0.0048643
7: -0.0145504, 0.0068375, -0.0154139, 0.0069756, -0.0215260, 0.0222514
8: -0.0055208, 0.0031305, -0.0081513, 0.0031737, -0.0086945, 0.0112818
9: -0.0135772, -0.0000236, -0.0136635, 0.0007340, -0.0143112, 0.0136399

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of NS_A1_B1_B2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_B2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0035686, upper bound: 0.0035084
time: 2.94 seconds

## Relational analysis of NS_A1_B1_B2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_B2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0034796, upper bound: 0.0035089
time: 3.06 seconds

## BFS NS instance: NS_A1_B2_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: 0.0058065, 0.0075183, 0.0058109, 0.0074456, -0.0016391, 0.0017074
1: -0.0009619, 0.0026444, -0.0009231, 0.0025036, -0.0034655, 0.0035675
2: -0.0044017, 0.0227064, -0.0032658, 0.0226000, -0.0270016, 0.0259722
3: -0.0045080, -0.0021194, -0.0045019, -0.0022209, -0.0022871, 0.0023825
4: -0.0005359, 0.0110530, -0.0005062, 0.0105608, -0.0110967, 0.0115592
5: -0.0021697, -0.0002017, -0.0020962, -0.0002309, -0.0019389, 0.0018946
6: 0.9904367, 0.9942325, 0.9905096, 0.9940977, -0.0036610, 0.0037228
7: -0.0144731, 0.0066251, -0.0144069, 0.0057341, -0.0202072, 0.0210319
8: -0.0052852, 0.0030639, -0.0050834, 0.0027848, -0.0080700, 0.0081473
9: -0.0134443, -0.0000915, -0.0128872, -0.0001496, -0.0132948, 0.0127957

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of NS_A1_B2_A2_B1_B1_A1_A1

### Relational analysis result of NS_A1_B2_A2_B1_B1_A1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0034287, upper bound: 0.0033857
time: 2.57 seconds

## Relational analysis of NS_A1_B2_A2_B1_B1_A1_A2

### Relational analysis result of NS_A1_B2_A2_B1_B1_A1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0032963, upper bound: 0.0033784
time: 2.25 seconds

## BFS NS instance: NS_A1_B2_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: 0.0057453, 0.0075245, 0.0058188, 0.0074335, -0.0016881, 0.0017057
1: -0.0015027, 0.0026564, -0.0008531, 0.0024800, -0.0039828, 0.0035094
2: -0.0044981, 0.0241925, -0.0030759, 0.0224075, -0.0269056, 0.0272684
3: -0.0045934, -0.0021108, -0.0044908, -0.0022378, -0.0023555, 0.0023800
4: -0.0009501, 0.0110948, -0.0004525, 0.0104785, -0.0114286, 0.0115474
5: -0.0021760, 0.0002061, -0.0020840, -0.0002837, -0.0018923, 0.0022901
6: 0.9894183, 0.9942440, 0.9906415, 0.9940752, -0.0046570, 0.0036025
7: -0.0153977, 0.0067007, -0.0142871, 0.0055851, -0.0209828, 0.0209878
8: -0.0081020, 0.0030876, -0.0047186, 0.0027381, -0.0108401, 0.0078063
9: -0.0134916, 0.0007198, -0.0127940, -0.0002546, -0.0132370, 0.0135138

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of NS_A1_B2_A2_B1_B1_A2_A1

### Relational analysis result of NS_A1_B2_A2_B1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0033840, upper bound: 0.0033827
time: 2.60 seconds

## Relational analysis of NS_A1_B2_A2_B1_B1_A2_A2

### Relational analysis result of NS_A1_B2_A2_B1_B1_A2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0032554, upper bound: 0.0033766
time: 2.43 seconds

## BFS NS instance: NS_A1_B2_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: 0.0058065, 0.0075183, 0.0057975, 0.0075360, -0.0017295, 0.0017208
1: -0.0009619, 0.0026444, -0.0010416, 0.0026786, -0.0036405, 0.0036860
2: -0.0044017, 0.0227064, -0.0046777, 0.0229255, -0.0273272, 0.0273841
3: -0.0045080, -0.0021194, -0.0045206, -0.0020948, -0.0024132, 0.0024012
4: -0.0005359, 0.0110530, -0.0005969, 0.0111727, -0.0117085, 0.0116500
5: -0.0021697, -0.0002017, -0.0021876, -0.0001416, -0.0020282, 0.0019859
6: 0.9904367, 0.9942325, 0.9902865, 0.9942653, -0.0038286, 0.0039459
7: -0.0144731, 0.0066251, -0.0146094, 0.0068416, -0.0213147, 0.0212345
8: -0.0052852, 0.0030639, -0.0057004, 0.0031318, -0.0084170, 0.0087643
9: -0.0134443, -0.0000915, -0.0135797, 0.0000281, -0.0134725, 0.0134883

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of NS_A1_B2_A2_B1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0034593, upper bound: 0.0035969
time: 2.58 seconds

## Relational analysis of NS_A1_B2_A2_B1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0034557, upper bound: 0.0035043
time: 2.41 seconds

## BFS NS instance: NS_A1_B2_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: 0.0057453, 0.0075245, 0.0058049, 0.0075242, -0.0017788, 0.0017196
1: -0.0015027, 0.0026564, -0.0009762, 0.0026557, -0.0041584, 0.0036325
2: -0.0044981, 0.0241925, -0.0044927, 0.0227457, -0.0272438, 0.0286852
3: -0.0045934, -0.0021108, -0.0045103, -0.0021113, -0.0024821, 0.0023994
4: -0.0009501, 0.0110948, -0.0005468, 0.0110925, -0.0120426, 0.0116416
5: -0.0021760, 0.0002061, -0.0021756, -0.0001909, -0.0019851, 0.0023817
6: 0.9894183, 0.9942440, 0.9904098, 0.9942433, -0.0048250, 0.0038342
7: -0.0153977, 0.0067007, -0.0144975, 0.0066964, -0.0220942, 0.0211982
8: -0.0081020, 0.0030876, -0.0053597, 0.0030863, -0.0111883, 0.0084473
9: -0.0134916, 0.0007198, -0.0134890, -0.0000700, -0.0134216, 0.0142087

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of NS_A1_B2_A2_B1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0034503, upper bound: 0.0036106
time: 2.88 seconds

## Relational analysis of NS_A1_B2_A2_B1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0034442, upper bound: 0.0035090
time: 2.26 seconds

## BFS NS instance: NS_A1_B2_A2_B2_B1_A1

### Backsubstitution after applying NS history:
0: 0.0058065, 0.0075183, 0.0057659, 0.0075358, -0.0017293, 0.0017525
1: -0.0009619, 0.0026444, -0.0013212, 0.0026783, -0.0036402, 0.0039656
2: -0.0044017, 0.0227064, -0.0046751, 0.0236936, -0.0280953, 0.0273816
3: -0.0045080, -0.0021194, -0.0045647, -0.0020950, -0.0024130, 0.0024453
4: -0.0005359, 0.0110530, -0.0008110, 0.0111716, -0.0117074, 0.0118641
5: -0.0021697, -0.0002017, -0.0021874, 0.0000692, -0.0022389, 0.0019858
6: 0.9904367, 0.9942325, 0.9897602, 0.9942650, -0.0038283, 0.0044723
7: -0.0144731, 0.0066251, -0.0150873, 0.0068396, -0.0213127, 0.0217124
8: -0.0052852, 0.0030639, -0.0071563, 0.0031311, -0.0084163, 0.0102202
9: -0.0134443, -0.0000915, -0.0135785, 0.0004474, -0.0138917, 0.0134870

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of NS_A1_B2_A2_B2_B1_A1_A1

### Relational analysis result of NS_A1_B2_A2_B2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0035616, upper bound: 0.0033863
time: 2.63 seconds

## Relational analysis of NS_A1_B2_A2_B2_B1_A1_A2

### Relational analysis result of NS_A1_B2_A2_B2_B1_A1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0034245, upper bound: 0.0033791
time: 2.50 seconds

## BFS NS instance: NS_A1_B2_A2_B2_B1_A2

### Backsubstitution after applying NS history:
0: 0.0057453, 0.0075245, 0.0057736, 0.0075237, -0.0017783, 0.0017509
1: -0.0015027, 0.0026564, -0.0012531, 0.0026548, -0.0041575, 0.0039095
2: -0.0044981, 0.0241925, -0.0044851, 0.0235065, -0.0280046, 0.0286776
3: -0.0045934, -0.0021108, -0.0045540, -0.0021120, -0.0024814, 0.0024431
4: -0.0009501, 0.0110948, -0.0007589, 0.0110892, -0.0120393, 0.0118537
5: -0.0021760, 0.0002061, -0.0021751, 0.0000179, -0.0021938, 0.0023812
6: 0.9894183, 0.9942440, 0.9898883, 0.9942424, -0.0048242, 0.0043557
7: -0.0153977, 0.0067007, -0.0149709, 0.0066905, -0.0220882, 0.0216716
8: -0.0081020, 0.0030876, -0.0068018, 0.0030844, -0.0111864, 0.0098894
9: -0.0134916, 0.0007198, -0.0134852, 0.0003453, -0.0138369, 0.0142050

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of NS_A1_B2_A2_B2_B1_A2_A1

### Relational analysis result of NS_A1_B2_A2_B2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0034917, upper bound: 0.0033829
time: 2.41 seconds

## Relational analysis of NS_A1_B2_A2_B2_B1_A2_A2

### Relational analysis result of NS_A1_B2_A2_B2_B1_A2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0033631, upper bound: 0.0033767
time: 2.67 seconds

## BFS NS instance: NS_A1_B2_A2_B2_B2_B1

### Backsubstitution after applying NS history:
0: 0.0057902, 0.0075591, 0.0057710, 0.0075806, -0.0017904, 0.0017881
1: -0.0011065, 0.0027233, -0.0012763, 0.0027649, -0.0038714, 0.0039995
2: -0.0050376, 0.0231037, -0.0053739, 0.0235702, -0.0286079, 0.0284776
3: -0.0045308, -0.0020626, -0.0045576, -0.0020326, -0.0024982, 0.0024950
4: -0.0006466, 0.0113286, -0.0007766, 0.0114743, -0.0121209, 0.0121053
5: -0.0022109, -0.0000926, -0.0022326, 0.0000354, -0.0022462, 0.0021400
6: 0.9901645, 0.9943079, 0.9898447, 0.9943478, -0.0041834, 0.0044633
7: -0.0147203, 0.0071239, -0.0150105, 0.0073877, -0.0221080, 0.0221345
8: -0.0060382, 0.0032202, -0.0069225, 0.0033029, -0.0093411, 0.0101427
9: -0.0137563, 0.0001254, -0.0139212, 0.0003801, -0.0141363, 0.0140466

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of NS_A1_B2_A2_B2_B2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0036045, upper bound: 0.0035250
time: 3.10 seconds

## Relational analysis of NS_A1_B2_A2_B2_B2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0035208, upper bound: 0.0035275
time: 2.21 seconds

## BFS NS instance: NS_A1_B2_A2_B2_B2_B2

### Backsubstitution after applying NS history:
0: 0.0057976, 0.0075472, 0.0057071, 0.0075882, -0.0017905, 0.0018401
1: -0.0010404, 0.0027004, -0.0018406, 0.0027797, -0.0038200, 0.0045410
2: -0.0048531, 0.0229221, -0.0054927, 0.0251208, -0.0299738, 0.0284148
3: -0.0045204, -0.0020791, -0.0046467, -0.0020220, -0.0024984, 0.0025676
4: -0.0005960, 0.0112487, -0.0012089, 0.0115258, -0.0121218, 0.0124575
5: -0.0021989, -0.0001425, -0.0022403, 0.0004608, -0.0026598, 0.0020978
6: 0.9902889, 0.9942860, 0.9887820, 0.9943619, -0.0040731, 0.0055040
7: -0.0146073, 0.0069792, -0.0159753, 0.0074809, -0.0220882, 0.0229544
8: -0.0056940, 0.0031749, -0.0098614, 0.0033321, -0.0090260, 0.0130363
9: -0.0136657, 0.0000263, -0.0139795, 0.0012265, -0.0148922, 0.0140057

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of NS_A1_B2_A2_B2_B2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0036190, upper bound: 0.0035084
time: 2.71 seconds

## Relational analysis of NS_A1_B2_A2_B2_B2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0035298, upper bound: 0.0035089
time: 2.94 seconds

## BFS NS instance: NS_A2_B1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: 0.0057670, 0.0075876, 0.0058434, 0.0074089, -0.0016419, 0.0017442
1: -0.0013109, 0.0027785, -0.0006360, 0.0024325, -0.0037434, 0.0034145
2: -0.0054831, 0.0236654, -0.0026927, 0.0218110, -0.0272942, 0.0263581
3: -0.0045631, -0.0020228, -0.0044566, -0.0022721, -0.0022910, 0.0024337
4: -0.0008032, 0.0115217, -0.0002863, 0.0103125, -0.0111157, 0.0118080
5: -0.0022397, 0.0000615, -0.0020592, -0.0004474, -0.0017923, 0.0021206
6: 0.9897795, 0.9943608, 0.9910504, 0.9940298, -0.0042503, 0.0033104
7: -0.0150697, 0.0074734, -0.0139160, 0.0052845, -0.0203543, 0.0213894
8: -0.0071028, 0.0033297, -0.0035880, 0.0026439, -0.0097468, 0.0069177
9: -0.0139748, 0.0004320, -0.0126061, -0.0005802, -0.0133945, 0.0130381

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of NS_A2_B1_B1_A2_B1_A1_A1

### Relational analysis result of NS_A2_B1_B1_A2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0034114, upper bound: 0.0034550
time: 2.54 seconds

## Relational analysis of NS_A2_B1_B1_A2_B1_A1_A2

### Relational analysis result of NS_A2_B1_B1_A2_B1_A1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0032687, upper bound: 0.0034505
time: 1.98 seconds

## BFS NS instance: NS_A2_B1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.0057031, 0.0075955, 0.0058509, 0.0073957, -0.0015196, 0.0017446
1: -0.0018757, 0.0027939, -0.0005852, 0.0024069, -0.0042826, 0.0030188
2: -0.0056073, 0.0252172, -0.0024858, 0.0216484, -0.0243494, 0.0277030
3: -0.0046523, -0.0020118, -0.0044461, -0.0022905, -0.0021204, 0.0024343
4: -0.0012357, 0.0115755, -0.0002355, 0.0102228, -0.0102879, 0.0118110
5: -0.0022477, 0.0004873, -0.0020458, -0.0004846, -0.0015751, 0.0025331
6: 0.9887160, 0.9943755, 0.9911419, 0.9940053, -0.0052893, 0.0028889
7: -0.0160353, 0.0075708, -0.0138091, 0.0051222, -0.0211575, 0.0191001
8: -0.0100442, 0.0033602, -0.0033379, 0.0025931, -0.0126373, 0.0059839
9: -0.0140357, 0.0012791, -0.0125046, -0.0006671, -0.0119431, 0.0137837

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of NS_A2_B1_B1_A2_B1_A2_A1

### Relational analysis result of NS_A2_B1_B1_A2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0033714, upper bound: 0.0034624
time: 1.68 seconds

## Relational analysis of NS_A2_B1_B1_A2_B1_A2_A2

### Relational analysis result of NS_A2_B1_B1_A2_B1_A2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0032362, upper bound: 0.0034590
time: 2.62 seconds

## BFS NS instance: NS_A2_B1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.0057670, 0.0075876, 0.0058265, 0.0075025, -0.0017355, 0.0017611
1: -0.0013109, 0.0027785, -0.0007855, 0.0026138, -0.0039247, 0.0035640
2: -0.0054831, 0.0236654, -0.0041547, 0.0222219, -0.0277050, 0.0278201
3: -0.0045631, -0.0020228, -0.0044802, -0.0021415, -0.0024216, 0.0024573
4: -0.0008032, 0.0115217, -0.0004008, 0.0109460, -0.0117492, 0.0119225
5: -0.0022397, 0.0000615, -0.0021537, -0.0003346, -0.0019051, 0.0022152
6: 0.9897795, 0.9943608, 0.9907687, 0.9942033, -0.0044239, 0.0035921
7: -0.0150697, 0.0074734, -0.0141716, 0.0064314, -0.0215011, 0.0216450
8: -0.0071028, 0.0033297, -0.0043668, 0.0030032, -0.0101061, 0.0076965
9: -0.0139748, 0.0004320, -0.0133232, -0.0003559, -0.0136188, 0.0137552

Time for backsubstitution: 2.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of NS_A2_B1_B1_A2_B2_A1_A1

### Relational analysis result of NS_A2_B1_B1_A2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0035267, upper bound: 0.0035687
time: 3.07 seconds

## Relational analysis of NS_A2_B1_B1_A2_B2_A1_A2

### Relational analysis result of NS_A2_B1_B1_A2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0034325, upper bound: 0.0035707
time: 2.76 seconds

## BFS NS instance: NS_A2_B1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.0057031, 0.0075955, 0.0058335, 0.0074906, -0.0017874, 0.0017620
1: -0.0018757, 0.0027939, -0.0007232, 0.0025907, -0.0044664, 0.0035171
2: -0.0056073, 0.0252172, -0.0039684, 0.0220506, -0.0276579, 0.0291856
3: -0.0046523, -0.0020118, -0.0044703, -0.0021581, -0.0024941, 0.0024586
4: -0.0012357, 0.0115755, -0.0003531, 0.0108653, -0.0121010, 0.0119286
5: -0.0022477, 0.0004873, -0.0021417, -0.0003816, -0.0018661, 0.0026290
6: 0.9887160, 0.9943755, 0.9908862, 0.9941810, -0.0054650, 0.0034893
7: -0.0160353, 0.0075708, -0.0140651, 0.0062852, -0.0223204, 0.0216359
8: -0.0100442, 0.0033602, -0.0040421, 0.0029574, -0.0130017, 0.0074024
9: -0.0140357, 0.0012791, -0.0132318, -0.0004494, -0.0135862, 0.0145109

Time for backsubstitution: 1.84 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 5.71 + 595.07 = 600.78 seconds
