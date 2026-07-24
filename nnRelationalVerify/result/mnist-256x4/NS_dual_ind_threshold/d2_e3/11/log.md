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
execution time: IAR + RelationalAnalysis = 1.87 + 3.80 = 5.67 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.0040994, upper bound: 0.0040994

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 214

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 80

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040587, upper bound: 0.0039893
time: 2.77 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040587, upper bound: 0.0040586
time: 2.30 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 5.31 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 5.31
Output dim: 6, lower bound: -0.0040587, upper bound: 0.0039893
NS_A2, status: Status.UNKNOWN, split count: 1, time: 5.31
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

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 96

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039759, upper bound: 0.0039477
time: 2.27 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039759, upper bound: 0.0039477
time: 2.74 seconds

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

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039893, upper bound: 0.0040586
time: 1.80 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039893, upper bound: 0.0040586
time: 1.89 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 5.47 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 5.47
Output dim: 6, lower bound: -0.0039759, upper bound: 0.0039477
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 5.47
Output dim: 6, lower bound: -0.0039759, upper bound: 0.0039477
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 5.47
Output dim: 6, lower bound: -0.0039893, upper bound: 0.0040586
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 5.47
Output dim: 6, lower bound: -0.0039893, upper bound: 0.0040586

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
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 214

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039429, upper bound: 0.0038412
time: 6.66 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039429, upper bound: 0.0039132
time: 3.11 seconds

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

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 214

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039888, upper bound: 0.0038412
time: 2.83 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039429, upper bound: 0.0039132
time: 3.30 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: 0.0057335, 0.0076757, 0.0057769, 0.0075913, -0.0018578, 0.0018988
1: -0.0016077, 0.0029493, -0.0012236, 0.0027857, -0.0043934, 0.0041729
2: -0.0068605, 0.0244810, -0.0055410, 0.0234257, -0.0302862, 0.0300220
3: -0.0046100, -0.0018998, -0.0045493, -0.0020177, -0.0025923, 0.0026495
4: -0.0010305, 0.0121186, -0.0007363, 0.0115468, -0.0125773, 0.0128549
5: -0.0023288, 0.0002853, -0.0022434, -0.0000043, -0.0023245, 0.0025287
6: 0.9892205, 0.9945242, 0.9899439, 0.9943677, -0.0051472, 0.0045804
7: -0.0155772, 0.0085539, -0.0149206, 0.0075188, -0.0230960, 0.0234744
8: -0.0086488, 0.0036682, -0.0066485, 0.0033439, -0.0119927, 0.0103167
9: -0.0146504, 0.0008773, -0.0140032, 0.0003012, -0.0149515, 0.0148804

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 214

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 96

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039478, upper bound: 0.0039758
time: 2.36 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039478, upper bound: 0.0040253
time: 2.48 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: 0.0057335, 0.0076757, 0.0057335, 0.0076757, -0.0019422, 0.0019422
1: -0.0016077, 0.0029493, -0.0016077, 0.0029493, -0.0045570, 0.0045570
2: -0.0068605, 0.0244810, -0.0068605, 0.0244810, -0.0313416, 0.0313416
3: -0.0046100, -0.0018998, -0.0046100, -0.0018998, -0.0027101, 0.0027101
4: -0.0010305, 0.0121186, -0.0010305, 0.0121186, -0.0131491, 0.0131491
5: -0.0023288, 0.0002853, -0.0023288, 0.0002853, -0.0026141, 0.0026141
6: 0.9892205, 0.9945242, 0.9892205, 0.9945242, -0.0053037, 0.0053037
7: -0.0155772, 0.0085539, -0.0155772, 0.0085539, -0.0241311, 0.0241311
8: -0.0086488, 0.0036682, -0.0086488, 0.0036682, -0.0123170, 0.0123170
9: -0.0146504, 0.0008773, -0.0146504, 0.0008773, -0.0155276, 0.0155276

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 214

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 96

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039478, upper bound: 0.0039758
time: 2.64 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039478, upper bound: 0.0040253
time: 2.88 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 7.34 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 7.34
Output dim: 6, lower bound: -0.0039429, upper bound: 0.0038412
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 7.34
Output dim: 6, lower bound: -0.0039429, upper bound: 0.0039132
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 7.34
Output dim: 6, lower bound: -0.0039888, upper bound: 0.0038412
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 7.34
Output dim: 6, lower bound: -0.0039429, upper bound: 0.0039132
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 7.34
Output dim: 6, lower bound: -0.0039478, upper bound: 0.0039758
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 7.34
Output dim: 6, lower bound: -0.0039478, upper bound: 0.0040253
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 7.34
Output dim: 6, lower bound: -0.0039478, upper bound: 0.0039758
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 7.34
Output dim: 6, lower bound: -0.0039478, upper bound: 0.0040253

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.0058060, 0.0074591, 0.0057867, 0.0075688, -0.0017628, 0.0016723
1: -0.0009667, 0.0025297, -0.0011368, 0.0027422, -0.0037089, 0.0036665
2: -0.0034762, 0.0227198, -0.0051903, 0.0231870, -0.0266632, 0.0279102
3: -0.0045088, -0.0022021, -0.0045356, -0.0020490, -0.0024598, 0.0023335
4: -0.0005396, 0.0106520, -0.0006698, 0.0113948, -0.0119344, 0.0113218
5: -0.0021099, -0.0001980, -0.0022207, -0.0000698, -0.0020401, 0.0020228
6: 0.9904276, 0.9941227, 0.9901074, 0.9943261, -0.0038985, 0.0040153
7: -0.0144814, 0.0058991, -0.0147721, 0.0072437, -0.0217251, 0.0206713
8: -0.0053106, 0.0028365, -0.0061961, 0.0032577, -0.0085684, 0.0090326
9: -0.0129904, -0.0000841, -0.0138312, 0.0001709, -0.0131613, 0.0137470

Time for backsubstitution: 2.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039018, upper bound: 0.0038412
time: 2.21 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039018, upper bound: 0.0038412
time: 2.04 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.0057926, 0.0075508, 0.0057773, 0.0076034, -0.0018108, 0.0017735
1: -0.0010850, 0.0027074, -0.0012202, 0.0028092, -0.0038942, 0.0039276
2: -0.0049096, 0.0230447, -0.0057310, 0.0234163, -0.0283259, 0.0287757
3: -0.0045274, -0.0020741, -0.0045488, -0.0020007, -0.0025267, 0.0024747
4: -0.0006302, 0.0112732, -0.0007337, 0.0116291, -0.0122592, 0.0120069
5: -0.0022026, -0.0001088, -0.0022557, -0.0000069, -0.0021957, 0.0021469
6: 0.9902049, 0.9942927, 0.9899504, 0.9943902, -0.0041854, 0.0043424
7: -0.0146836, 0.0070235, -0.0149147, 0.0076678, -0.0223514, 0.0219383
8: -0.0059264, 0.0031888, -0.0066306, 0.0033906, -0.0093170, 0.0098193
9: -0.0136935, 0.0000932, -0.0140964, 0.0002960, -0.0139895, 0.0141895

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039018, upper bound: 0.0039132
time: 1.98 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039018, upper bound: 0.0039132
time: 2.97 seconds

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

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039018, upper bound: 0.0038412
time: 2.57 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039018, upper bound: 0.0038412
time: 2.95 seconds

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

Time for backsubstitution: 2.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039237, upper bound: 0.0039132
time: 2.94 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039237, upper bound: 0.0039132
time: 3.10 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: 0.0057811, 0.0076072, 0.0057828, 0.0075730, -0.0017919, 0.0018244
1: -0.0011868, 0.0028165, -0.0011716, 0.0027503, -0.0039370, 0.0039881
2: -0.0057897, 0.0233243, -0.0052553, 0.0232827, -0.0290724, 0.0285797
3: -0.0045435, -0.0019955, -0.0045411, -0.0020432, -0.0025003, 0.0025456
4: -0.0007081, 0.0116545, -0.0006965, 0.0114230, -0.0121311, 0.0123510
5: -0.0022595, -0.0000321, -0.0022249, -0.0000435, -0.0022160, 0.0021928
6: 0.9900132, 0.9943972, 0.9900417, 0.9943338, -0.0043206, 0.0043554
7: -0.0148575, 0.0077138, -0.0148316, 0.0072947, -0.0221523, 0.0225455
8: -0.0064564, 0.0034050, -0.0063775, 0.0032737, -0.0097301, 0.0097825
9: -0.0141251, 0.0002458, -0.0138631, 0.0002231, -0.0143482, 0.0141089

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038412, upper bound: 0.0039429
time: 2.09 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039132, upper bound: 0.0039429
time: 2.89 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.0057430, 0.0076496, 0.0057790, 0.0075848, -0.0018418, 0.0018706
1: -0.0015230, 0.0028987, -0.0012053, 0.0027732, -0.0042962, 0.0041040
2: -0.0064526, 0.0242482, -0.0054405, 0.0233753, -0.0298280, 0.0296887
3: -0.0045966, -0.0019363, -0.0045464, -0.0020267, -0.0025699, 0.0026102
4: -0.0009656, 0.0119418, -0.0007223, 0.0115032, -0.0124688, 0.0126641
5: -0.0023024, 0.0002214, -0.0022369, -0.0000181, -0.0022843, 0.0024583
6: 0.9893801, 0.9944759, 0.9899783, 0.9943558, -0.0049757, 0.0044976
7: -0.0154324, 0.0082339, -0.0148893, 0.0074400, -0.0228723, 0.0231232
8: -0.0082075, 0.0035680, -0.0065530, 0.0033192, -0.0115267, 0.0101210
9: -0.0144503, 0.0007501, -0.0139539, 0.0002737, -0.0147240, 0.0147040

Time for backsubstitution: 2.12 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038412, upper bound: 0.0039888
time: 2.44 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039132, upper bound: 0.0039888
time: 2.80 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.0057811, 0.0076072, 0.0057391, 0.0076563, -0.0018752, 0.0018680
1: -0.0011868, 0.0028165, -0.0015577, 0.0029116, -0.0040984, 0.0043742
2: -0.0057897, 0.0233243, -0.0065571, 0.0243434, -0.0301331, 0.0298814
3: -0.0045435, -0.0019955, -0.0046021, -0.0019269, -0.0026166, 0.0026066
4: -0.0007081, 0.0116545, -0.0009922, 0.0119871, -0.0126952, 0.0126467
5: -0.0022595, -0.0000321, -0.0023092, 0.0002475, -0.0025070, 0.0022770
6: 0.9900132, 0.9943972, 0.9893148, 0.9944882, -0.0044749, 0.0050824
7: -0.0148575, 0.0077138, -0.0154916, 0.0083158, -0.0231733, 0.0232054
8: -0.0064564, 0.0034050, -0.0083880, 0.0035936, -0.0100500, 0.0117930
9: -0.0141251, 0.0002458, -0.0145015, 0.0008021, -0.0149273, 0.0147474

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038412, upper bound: 0.0039429
time: 2.56 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038412, upper bound: 0.0039429
time: 2.77 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.0057430, 0.0076496, 0.0057356, 0.0076701, -0.0019270, 0.0019140
1: -0.0015230, 0.0028987, -0.0015892, 0.0029383, -0.0044614, 0.0044879
2: -0.0064526, 0.0242482, -0.0067724, 0.0244300, -0.0308826, 0.0310206
3: -0.0045966, -0.0019363, -0.0046070, -0.0019077, -0.0026889, 0.0026708
4: -0.0009656, 0.0119418, -0.0010163, 0.0120804, -0.0130460, 0.0129581
5: -0.0023024, 0.0002214, -0.0023231, 0.0002713, -0.0025737, 0.0025445
6: 0.9893801, 0.9944759, 0.9892555, 0.9945138, -0.0051337, 0.0052204
7: -0.0154324, 0.0082339, -0.0155455, 0.0084847, -0.0239171, 0.0237793
8: -0.0082075, 0.0035680, -0.0085520, 0.0036465, -0.0118541, 0.0121200
9: -0.0144503, 0.0007501, -0.0146072, 0.0008494, -0.0152997, 0.0153573

Time for backsubstitution: 2.12 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038412, upper bound: 0.0039888
time: 2.41 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038412, upper bound: 0.0039888
time: 3.00 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 7.77 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 7.77
Output dim: 6, lower bound: -0.0039018, upper bound: 0.0038412
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 7.77
Output dim: 6, lower bound: -0.0039018, upper bound: 0.0038412
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 7.77
Output dim: 6, lower bound: -0.0039018, upper bound: 0.0039132
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 7.77
Output dim: 6, lower bound: -0.0039018, upper bound: 0.0039132
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 7.77
Output dim: 6, lower bound: -0.0039018, upper bound: 0.0038412
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 7.77
Output dim: 6, lower bound: -0.0039018, upper bound: 0.0038412
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 7.77
Output dim: 6, lower bound: -0.0039237, upper bound: 0.0039132
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 7.77
Output dim: 6, lower bound: -0.0039237, upper bound: 0.0039132
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 7.77
Output dim: 6, lower bound: -0.0038412, upper bound: 0.0039429
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 7.77
Output dim: 6, lower bound: -0.0039132, upper bound: 0.0039429
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 7.77
Output dim: 6, lower bound: -0.0038412, upper bound: 0.0039888
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 7.77
Output dim: 6, lower bound: -0.0039132, upper bound: 0.0039888
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 7.77
Output dim: 6, lower bound: -0.0038412, upper bound: 0.0039429
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 7.77
Output dim: 6, lower bound: -0.0038412, upper bound: 0.0039429
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 7.77
Output dim: 6, lower bound: -0.0038412, upper bound: 0.0039888
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 7.77
Output dim: 6, lower bound: -0.0038412, upper bound: 0.0039888

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.0058060, 0.0074591, 0.0058254, 0.0074905, -0.0016845, 0.0016337
1: -0.0009667, 0.0025297, -0.0007951, 0.0025906, -0.0035573, 0.0033248
2: -0.0034762, 0.0227198, -0.0039673, 0.0222482, -0.0257244, 0.0266871
3: -0.0045088, -0.0022021, -0.0044817, -0.0021582, -0.0023505, 0.0022796
4: -0.0005396, 0.0106520, -0.0004081, 0.0108648, -0.0114044, 0.0110601
5: -0.0021099, -0.0001980, -0.0021416, -0.0003274, -0.0017825, 0.0019436
6: 0.9904276, 0.9941227, 0.9907507, 0.9941810, -0.0037534, 0.0033720
7: -0.0144814, 0.0058991, -0.0141880, 0.0062843, -0.0207658, 0.0200871
8: -0.0053106, 0.0028365, -0.0044166, 0.0029572, -0.0082678, 0.0072531
9: -0.0129904, -0.0000841, -0.0132313, -0.0003416, -0.0126488, 0.0131471

Time for backsubstitution: 2.16 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 214

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037930, upper bound: 0.0036772
time: 2.37 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037511, upper bound: 0.0036690
time: 2.76 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.0058060, 0.0074591, 0.0057912, 0.0075716, -0.0017657, 0.0016679
1: -0.0009667, 0.0025297, -0.0010976, 0.0027477, -0.0037145, 0.0036273
2: -0.0034762, 0.0227198, -0.0052348, 0.0230793, -0.0265555, 0.0279546
3: -0.0045088, -0.0022021, -0.0045294, -0.0020450, -0.0024638, 0.0023273
4: -0.0005396, 0.0106520, -0.0006398, 0.0114141, -0.0119537, 0.0112918
5: -0.0021099, -0.0001980, -0.0022236, -0.0000993, -0.0020105, 0.0020256
6: 0.9904276, 0.9941227, 0.9901811, 0.9943314, -0.0039038, 0.0039416
7: -0.0144814, 0.0058991, -0.0147051, 0.0072786, -0.0217600, 0.0206042
8: -0.0053106, 0.0028365, -0.0059920, 0.0032687, -0.0085793, 0.0088285
9: -0.0129904, -0.0000841, -0.0138530, 0.0001121, -0.0131025, 0.0137688

Time for backsubstitution: 2.16 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 214

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037930, upper bound: 0.0036772
time: 2.34 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037511, upper bound: 0.0036690
time: 2.95 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.0057926, 0.0075508, 0.0058162, 0.0075261, -0.0017335, 0.0017346
1: -0.0010850, 0.0027074, -0.0008761, 0.0026594, -0.0037444, 0.0035835
2: -0.0049096, 0.0230447, -0.0045224, 0.0224709, -0.0273805, 0.0275671
3: -0.0045274, -0.0020741, -0.0044945, -0.0021087, -0.0024188, 0.0024204
4: -0.0006302, 0.0112732, -0.0004702, 0.0111054, -0.0117355, 0.0117434
5: -0.0022026, -0.0001088, -0.0021775, -0.0002663, -0.0019363, 0.0020687
6: 0.9902049, 0.9942927, 0.9905981, 0.9942469, -0.0040420, 0.0036947
7: -0.0146836, 0.0070235, -0.0143265, 0.0067198, -0.0214034, 0.0213501
8: -0.0059264, 0.0031888, -0.0048387, 0.0030936, -0.0090200, 0.0080275
9: -0.0136935, 0.0000932, -0.0135036, -0.0002200, -0.0134734, 0.0135968

Time for backsubstitution: 2.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 214

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 96

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039018, upper bound: 0.0038998
time: 3.01 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039018, upper bound: 0.0039132
time: 2.17 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.0057926, 0.0075508, 0.0057817, 0.0076057, -0.0018131, 0.0017692
1: -0.0010850, 0.0027074, -0.0011816, 0.0028137, -0.0038987, 0.0038890
2: -0.0049096, 0.0230447, -0.0057670, 0.0233100, -0.0282197, 0.0288117
3: -0.0045274, -0.0020741, -0.0045427, -0.0019975, -0.0025299, 0.0024686
4: -0.0006302, 0.0112732, -0.0007041, 0.0116447, -0.0122748, 0.0119773
5: -0.0022026, -0.0001088, -0.0022580, -0.0000360, -0.0021665, 0.0021492
6: 0.9902049, 0.9942927, 0.9900231, 0.9943945, -0.0041897, 0.0042697
7: -0.0146836, 0.0070235, -0.0148486, 0.0076961, -0.0223796, 0.0218722
8: -0.0059264, 0.0031888, -0.0064293, 0.0033995, -0.0093258, 0.0096180
9: -0.0136935, 0.0000932, -0.0141140, 0.0002380, -0.0139315, 0.0142072

Time for backsubstitution: 2.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 214

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 96

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039018, upper bound: 0.0038998
time: 2.97 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039018, upper bound: 0.0039132
time: 2.79 seconds

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

Time for backsubstitution: 2.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 214

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 96

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039018, upper bound: 0.0038182
time: 2.62 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039018, upper bound: 0.0038412
time: 2.76 seconds

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

Time for backsubstitution: 2.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 214

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 96

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039018, upper bound: 0.0038182
time: 2.28 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039018, upper bound: 0.0038412
time: 2.22 seconds

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

Time for backsubstitution: 2.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 214

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 96

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039018, upper bound: 0.0038975
time: 2.69 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039018, upper bound: 0.0039132
time: 2.88 seconds

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

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 214

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 96

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039018, upper bound: 0.0038975
time: 3.03 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039018, upper bound: 0.0039132
time: 2.62 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.0057912, 0.0075716, 0.0058060, 0.0074591, -0.0016679, 0.0017657
1: -0.0010976, 0.0027477, -0.0009667, 0.0025297, -0.0036273, 0.0037145
2: -0.0052348, 0.0230793, -0.0034762, 0.0227198, -0.0279546, 0.0265555
3: -0.0045294, -0.0020450, -0.0045088, -0.0022021, -0.0023273, 0.0024638
4: -0.0006398, 0.0114141, -0.0005396, 0.0106520, -0.0112918, 0.0119537
5: -0.0022236, -0.0000993, -0.0021099, -0.0001980, -0.0020256, 0.0020105
6: 0.9901811, 0.9943314, 0.9904276, 0.9941227, -0.0039416, 0.0039038
7: -0.0147051, 0.0072786, -0.0144814, 0.0058991, -0.0206042, 0.0217600
8: -0.0059920, 0.0032687, -0.0053106, 0.0028365, -0.0088285, 0.0085793
9: -0.0138530, 0.0001121, -0.0129904, -0.0000841, -0.0137688, 0.0131025

Time for backsubstitution: 2.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 214

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036987, upper bound: 0.0037857
time: 2.31 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036691, upper bound: 0.0038012
time: 2.41 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.0057817, 0.0076057, 0.0057926, 0.0075508, -0.0017692, 0.0018131
1: -0.0011816, 0.0028137, -0.0010850, 0.0027074, -0.0038890, 0.0038987
2: -0.0057670, 0.0233100, -0.0049096, 0.0230447, -0.0288117, 0.0282197
3: -0.0045427, -0.0019975, -0.0045274, -0.0020741, -0.0024686, 0.0025299
4: -0.0007041, 0.0116447, -0.0006302, 0.0112732, -0.0119773, 0.0122748
5: -0.0022580, -0.0000360, -0.0022026, -0.0001088, -0.0021492, 0.0021665
6: 0.9900231, 0.9943945, 0.9902049, 0.9942927, -0.0042697, 0.0041897
7: -0.0148486, 0.0076961, -0.0146836, 0.0070235, -0.0218722, 0.0223796
8: -0.0064293, 0.0033995, -0.0059264, 0.0031888, -0.0096180, 0.0093258
9: -0.0141140, 0.0002380, -0.0136935, 0.0000932, -0.0142072, 0.0139315

Time for backsubstitution: 2.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 214

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039133, upper bound: 0.0038559
time: 2.62 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039133, upper bound: 0.0039429
time: 2.71 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.0057534, 0.0076143, 0.0058024, 0.0074716, -0.0017182, 0.0018120
1: -0.0014316, 0.0028304, -0.0009986, 0.0025539, -0.0039855, 0.0038290
2: -0.0059015, 0.0239970, -0.0036715, 0.0228074, -0.0287090, 0.0276685
3: -0.0045822, -0.0019855, -0.0045138, -0.0021846, -0.0023975, 0.0025283
4: -0.0008956, 0.0117030, -0.0005640, 0.0107367, -0.0116323, 0.0122670
5: -0.0022667, 0.0001525, -0.0021225, -0.0001740, -0.0020928, 0.0022750
6: 0.9895521, 0.9944105, 0.9903675, 0.9941459, -0.0045937, 0.0040430
7: -0.0152761, 0.0078016, -0.0145359, 0.0060523, -0.0213284, 0.0223375
8: -0.0077314, 0.0034325, -0.0054766, 0.0028845, -0.0106159, 0.0089091
9: -0.0141800, 0.0006130, -0.0130862, -0.0000363, -0.0141437, 0.0136992

Time for backsubstitution: 2.16 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 214

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036987, upper bound: 0.0038232
time: 2.48 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036691, upper bound: 0.0038502
time: 2.14 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.0057436, 0.0076481, 0.0057888, 0.0075625, -0.0018188, 0.0018593
1: -0.0015177, 0.0028958, -0.0011183, 0.0027299, -0.0042476, 0.0040141
2: -0.0064296, 0.0242336, -0.0050914, 0.0231362, -0.0295658, 0.0293249
3: -0.0045957, -0.0019383, -0.0045327, -0.0020578, -0.0025379, 0.0025944
4: -0.0009616, 0.0119318, -0.0006557, 0.0113519, -0.0123135, 0.0125875
5: -0.0023009, 0.0002174, -0.0022143, -0.0000837, -0.0022172, 0.0024317
6: 0.9893900, 0.9944730, 0.9901422, 0.9943143, -0.0049243, 0.0043309
7: -0.0154233, 0.0082158, -0.0147405, 0.0071661, -0.0225893, 0.0229563
8: -0.0081798, 0.0035623, -0.0060998, 0.0032334, -0.0114132, 0.0096621
9: -0.0144390, 0.0007422, -0.0137826, 0.0001431, -0.0145821, 0.0145248

Time for backsubstitution: 2.17 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 214

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039133, upper bound: 0.0039111
time: 2.87 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039133, upper bound: 0.0039888
time: 6.74 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.0057912, 0.0075716, 0.0057606, 0.0075450, -0.0017539, 0.0018110
1: -0.0010976, 0.0027477, -0.0013678, 0.0026961, -0.0037937, 0.0041155
2: -0.0052348, 0.0230793, -0.0048187, 0.0238218, -0.0290566, 0.0278980
3: -0.0045294, -0.0020450, -0.0045721, -0.0020822, -0.0024472, 0.0025271
4: -0.0006398, 0.0114141, -0.0008468, 0.0112337, -0.0118735, 0.0122608
5: -0.0022236, -0.0000993, -0.0021967, 0.0001044, -0.0023280, 0.0020974
6: 0.9901811, 0.9943314, 0.9896723, 0.9942820, -0.0041009, 0.0046591
7: -0.0147051, 0.0072786, -0.0151671, 0.0069522, -0.0216573, 0.0224457
8: -0.0059920, 0.0032687, -0.0073994, 0.0031664, -0.0091584, 0.0106680
9: -0.0138530, 0.0001121, -0.0136489, 0.0005174, -0.0143704, 0.0137610

Time for backsubstitution: 2.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 214

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036994, upper bound: 0.0037857
time: 3.04 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036734, upper bound: 0.0038012
time: 2.69 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.0057817, 0.0076057, 0.0057491, 0.0076331, -0.0018515, 0.0018566
1: -0.0011816, 0.0028137, -0.0014699, 0.0028668, -0.0040483, 0.0042836
2: -0.0057670, 0.0233100, -0.0061952, 0.0241023, -0.0298693, 0.0295052
3: -0.0045427, -0.0019975, -0.0045882, -0.0019592, -0.0025834, 0.0025907
4: -0.0007041, 0.0116447, -0.0009250, 0.0118303, -0.0125344, 0.0125696
5: -0.0022580, -0.0000360, -0.0022857, 0.0001814, -0.0024394, 0.0022497
6: 0.9900231, 0.9943945, 0.9894801, 0.9944453, -0.0044222, 0.0049144
7: -0.0148486, 0.0076961, -0.0153416, 0.0080320, -0.0228806, 0.0230376
8: -0.0064293, 0.0033995, -0.0079309, 0.0035047, -0.0099340, 0.0113304
9: -0.0141140, 0.0002380, -0.0143240, 0.0006705, -0.0147845, 0.0145621

Time for backsubstitution: 2.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 214

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039133, upper bound: 0.0038559
time: 2.87 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039133, upper bound: 0.0039429
time: 2.90 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.0057534, 0.0076143, 0.0057572, 0.0075594, -0.0018060, 0.0018571
1: -0.0014316, 0.0028304, -0.0013975, 0.0027240, -0.0041556, 0.0042279
2: -0.0059015, 0.0239970, -0.0050434, 0.0239035, -0.0298050, 0.0290404
3: -0.0045822, -0.0019855, -0.0045768, -0.0020621, -0.0025200, 0.0025913
4: -0.0008956, 0.0117030, -0.0008695, 0.0113312, -0.0122268, 0.0125725
5: -0.0022667, 0.0001525, -0.0022112, 0.0001268, -0.0023936, 0.0023637
6: 0.9895521, 0.9944105, 0.9896164, 0.9943086, -0.0047565, 0.0047941
7: -0.0152761, 0.0078016, -0.0152179, 0.0071285, -0.0224045, 0.0230195
8: -0.0077314, 0.0034325, -0.0075541, 0.0032216, -0.0109530, 0.0109866
9: -0.0141800, 0.0006130, -0.0137591, 0.0005620, -0.0147420, 0.0143721

Time for backsubstitution: 2.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 214

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036993, upper bound: 0.0038232
time: 2.52 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036734, upper bound: 0.0038501
time: 3.07 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.0057436, 0.0076481, 0.0057455, 0.0076467, -0.0019030, 0.0019026
1: -0.0015177, 0.0028958, -0.0015011, 0.0028930, -0.0044107, 0.0043969
2: -0.0064296, 0.0242336, -0.0064065, 0.0241880, -0.0306176, 0.0306400
3: -0.0045957, -0.0019383, -0.0045931, -0.0019404, -0.0026554, 0.0026548
4: -0.0009616, 0.0119318, -0.0009488, 0.0119218, -0.0128834, 0.0128807
5: -0.0023009, 0.0002174, -0.0022994, 0.0002049, -0.0025058, 0.0025168
6: 0.9893900, 0.9944730, 0.9894214, 0.9944704, -0.0050804, 0.0050517
7: -0.0154233, 0.0082158, -0.0153949, 0.0081977, -0.0236209, 0.0236107
8: -0.0081798, 0.0035623, -0.0080934, 0.0035566, -0.0117364, 0.0116557
9: -0.0144390, 0.0007422, -0.0144277, 0.0007173, -0.0151563, 0.0151698

Time for backsubstitution: 2.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 214

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039133, upper bound: 0.0039111
time: 2.72 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039133, upper bound: 0.0039888
time: 3.13 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 8.28 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 8.28
Output dim: 6, lower bound: -0.0037930, upper bound: 0.0036772
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 8.28
Output dim: 6, lower bound: -0.0037511, upper bound: 0.0036690
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 8.28
Output dim: 6, lower bound: -0.0037930, upper bound: 0.0036772
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 8.28
Output dim: 6, lower bound: -0.0037511, upper bound: 0.0036690
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 8.28
Output dim: 6, lower bound: -0.0039018, upper bound: 0.0038998
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 8.28
Output dim: 6, lower bound: -0.0039018, upper bound: 0.0039132
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 8.28
Output dim: 6, lower bound: -0.0039018, upper bound: 0.0038998
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 8.28
Output dim: 6, lower bound: -0.0039018, upper bound: 0.0039132
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 8.28
Output dim: 6, lower bound: -0.0039018, upper bound: 0.0038182
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 8.28
Output dim: 6, lower bound: -0.0039018, upper bound: 0.0038412
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 8.28
Output dim: 6, lower bound: -0.0039018, upper bound: 0.0038182
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 8.28
Output dim: 6, lower bound: -0.0039018, upper bound: 0.0038412
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 8.28
Output dim: 6, lower bound: -0.0039018, upper bound: 0.0038975
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 8.28
Output dim: 6, lower bound: -0.0039018, upper bound: 0.0039132
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 8.28
Output dim: 6, lower bound: -0.0039018, upper bound: 0.0038975
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 8.28
Output dim: 6, lower bound: -0.0039018, upper bound: 0.0039132
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 8.28
Output dim: 6, lower bound: -0.0036987, upper bound: 0.0037857
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 8.28
Output dim: 6, lower bound: -0.0036691, upper bound: 0.0038012
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 8.28
Output dim: 6, lower bound: -0.0039133, upper bound: 0.0038559
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 8.28
Output dim: 6, lower bound: -0.0039133, upper bound: 0.0039429
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 8.28
Output dim: 6, lower bound: -0.0036987, upper bound: 0.0038232
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 8.28
Output dim: 6, lower bound: -0.0036691, upper bound: 0.0038502
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 8.28
Output dim: 6, lower bound: -0.0039133, upper bound: 0.0039111
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 8.28
Output dim: 6, lower bound: -0.0039133, upper bound: 0.0039888
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 8.28
Output dim: 6, lower bound: -0.0036994, upper bound: 0.0037857
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 8.28
Output dim: 6, lower bound: -0.0036734, upper bound: 0.0038012
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 8.28
Output dim: 6, lower bound: -0.0039133, upper bound: 0.0038559
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 8.28
Output dim: 6, lower bound: -0.0039133, upper bound: 0.0039429
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 8.28
Output dim: 6, lower bound: -0.0036993, upper bound: 0.0038232
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 8.28
Output dim: 6, lower bound: -0.0036734, upper bound: 0.0038501
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 8.28
Output dim: 6, lower bound: -0.0039133, upper bound: 0.0039111
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 8.28
Output dim: 6, lower bound: -0.0039133, upper bound: 0.0039888

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.0058243, 0.0074172, 0.0058267, 0.0074872, -0.0016629, 0.0015904
1: -0.0008045, 0.0024484, -0.0007831, 0.0025841, -0.0033887, 0.0032316
2: -0.0028210, 0.0222741, -0.0039156, 0.0222154, -0.0250363, 0.0261897
3: -0.0044832, -0.0022606, -0.0044798, -0.0021629, -0.0023203, 0.0022192
4: -0.0004153, 0.0103681, -0.0003990, 0.0108424, -0.0112577, 0.0107670
5: -0.0020675, -0.0003203, -0.0021383, -0.0003364, -0.0017311, 0.0018180
6: 0.9907330, 0.9940450, 0.9907733, 0.9941748, -0.0034419, 0.0032717
7: -0.0142041, 0.0053851, -0.0141676, 0.0062438, -0.0204479, 0.0195527
8: -0.0044657, 0.0026755, -0.0043544, 0.0029445, -0.0074102, 0.0070299
9: -0.0126690, -0.0003275, -0.0132059, -0.0003595, -0.0123095, 0.0128785

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0034326, upper bound: 0.0033983
time: 2.67 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0034366, upper bound: 0.0032868
time: 2.89 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.0057647, 0.0074190, 0.0058338, 0.0074745, -0.0017099, 0.0015852
1: -0.0013318, 0.0024521, -0.0007204, 0.0025596, -0.0038914, 0.0031725
2: -0.0028504, 0.0237228, -0.0037174, 0.0220430, -0.0248934, 0.0274403
3: -0.0045664, -0.0022580, -0.0044699, -0.0021806, -0.0023859, 0.0022119
4: -0.0008192, 0.0103808, -0.0003509, 0.0107565, -0.0115757, 0.0107318
5: -0.0020694, 0.0000772, -0.0021255, -0.0003837, -0.0016857, 0.0022027
6: 0.9897402, 0.9940485, 0.9908913, 0.9941514, -0.0044112, 0.0031571
7: -0.0151055, 0.0054082, -0.0140603, 0.0060883, -0.0211938, 0.0194686
8: -0.0072117, 0.0026827, -0.0040278, 0.0028958, -0.0101075, 0.0067105
9: -0.0126835, 0.0004634, -0.0131087, -0.0004536, -0.0122299, 0.0135721

Time for backsubstitution: 2.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0033683, upper bound: 0.0033840
time: 2.72 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0033633, upper bound: 0.0032554
time: 2.87 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.0058243, 0.0074172, 0.0057926, 0.0075682, -0.0017439, 0.0016246
1: -0.0008045, 0.0024484, -0.0010853, 0.0027410, -0.0035456, 0.0035338
2: -0.0028210, 0.0222741, -0.0051811, 0.0230456, -0.0258666, 0.0274552
3: -0.0044832, -0.0022606, -0.0045275, -0.0020498, -0.0024333, 0.0022669
4: -0.0004153, 0.0103681, -0.0006304, 0.0113908, -0.0118061, 0.0109985
5: -0.0020675, -0.0003203, -0.0022201, -0.0001086, -0.0019589, 0.0018998
6: 0.9907330, 0.9940450, 0.9902043, 0.9943250, -0.0035920, 0.0038407
7: -0.0142041, 0.0053851, -0.0146841, 0.0072365, -0.0214406, 0.0200693
8: -0.0044657, 0.0026755, -0.0059281, 0.0032555, -0.0077212, 0.0086036
9: -0.0126690, -0.0003275, -0.0138266, 0.0000937, -0.0127627, 0.0134992

Time for backsubstitution: 2.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0034928, upper bound: 0.0033983
time: 2.95 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0034999, upper bound: 0.0032868
time: 2.93 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.0057647, 0.0074190, 0.0057996, 0.0075562, -0.0017916, 0.0016194
1: -0.0013318, 0.0024521, -0.0010228, 0.0027178, -0.0040497, 0.0034749
2: -0.0028504, 0.0237228, -0.0049939, 0.0228740, -0.0257244, 0.0287168
3: -0.0045664, -0.0022580, -0.0045176, -0.0020665, -0.0024999, 0.0022596
4: -0.0008192, 0.0103808, -0.0005826, 0.0113097, -0.0121289, 0.0109634
5: -0.0020694, 0.0000772, -0.0022080, -0.0001557, -0.0019137, 0.0022853
6: 0.9897402, 0.9940485, 0.9903219, 0.9943027, -0.0045626, 0.0037266
7: -0.0151055, 0.0054082, -0.0145773, 0.0070897, -0.0221951, 0.0199856
8: -0.0072117, 0.0026827, -0.0056027, 0.0032095, -0.0104212, 0.0082854
9: -0.0126835, 0.0004634, -0.0137348, -0.0000000, -0.0126835, 0.0141982

Time for backsubstitution: 2.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0034162, upper bound: 0.0033840
time: 2.72 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0034137, upper bound: 0.0032554
time: 2.98 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: 0.0058251, 0.0075060, 0.0058162, 0.0075261, -0.0017009, 0.0016897
1: -0.0007975, 0.0026204, -0.0008761, 0.0026594, -0.0034568, 0.0034966
2: -0.0042082, 0.0222547, -0.0045224, 0.0224709, -0.0266791, 0.0267771
3: -0.0044821, -0.0021367, -0.0044945, -0.0021087, -0.0023734, 0.0023578
4: -0.0004099, 0.0109692, -0.0004702, 0.0111054, -0.0115153, 0.0114394
5: -0.0021572, -0.0003256, -0.0021775, -0.0002663, -0.0018909, 0.0018519
6: 0.9907463, 0.9942095, 0.9905981, 0.9942469, -0.0035006, 0.0036114
7: -0.0141920, 0.0064733, -0.0143265, 0.0067198, -0.0209118, 0.0207999
8: -0.0044290, 0.0030164, -0.0048387, 0.0030936, -0.0075226, 0.0078551
9: -0.0133495, -0.0003381, -0.0135036, -0.0002200, -0.0131294, 0.0131655

Time for backsubstitution: 2.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038212, upper bound: 0.0039072
time: 3.19 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038212, upper bound: 0.0039072
time: 2.46 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.0057962, 0.0075394, 0.0058162, 0.0075261, -0.0017299, 0.0017232
1: -0.0010534, 0.0026853, -0.0008761, 0.0026594, -0.0037128, 0.0035615
2: -0.0047316, 0.0229580, -0.0045224, 0.0224709, -0.0272025, 0.0274804
3: -0.0045225, -0.0020900, -0.0044945, -0.0021087, -0.0024138, 0.0024045
4: -0.0006060, 0.0111960, -0.0004702, 0.0111054, -0.0117114, 0.0116662
5: -0.0021911, -0.0001326, -0.0021775, -0.0002663, -0.0019248, 0.0020449
6: 0.9902643, 0.9942716, 0.9905981, 0.9942469, -0.0039826, 0.0036736
7: -0.0146296, 0.0068839, -0.0143265, 0.0067198, -0.0213494, 0.0212104
8: -0.0057620, 0.0031450, -0.0048387, 0.0030936, -0.0088556, 0.0079837
9: -0.0136062, 0.0000459, -0.0135036, -0.0002200, -0.0133861, 0.0135494

Time for backsubstitution: 2.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038212, upper bound: 0.0039237
time: 2.91 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038212, upper bound: 0.0039237
time: 2.21 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.0058251, 0.0075060, 0.0057817, 0.0076057, -0.0017806, 0.0017243
1: -0.0007975, 0.0026204, -0.0011816, 0.0028137, -0.0036111, 0.0038020
2: -0.0042082, 0.0222547, -0.0057670, 0.0233100, -0.0275183, 0.0280217
3: -0.0044821, -0.0021367, -0.0045427, -0.0019975, -0.0024846, 0.0024060
4: -0.0004099, 0.0109692, -0.0007041, 0.0116447, -0.0120546, 0.0116733
5: -0.0021572, -0.0003256, -0.0022580, -0.0000360, -0.0021212, 0.0019324
6: 0.9907463, 0.9942095, 0.9900231, 0.9943945, -0.0036483, 0.0041865
7: -0.0141920, 0.0064733, -0.0148486, 0.0076961, -0.0218881, 0.0213220
8: -0.0044290, 0.0030164, -0.0064293, 0.0033995, -0.0078284, 0.0094457
9: -0.0133495, -0.0003381, -0.0141140, 0.0002380, -0.0135875, 0.0137760

Time for backsubstitution: 2.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038211, upper bound: 0.0038998
time: 2.74 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038559, upper bound: 0.0038998
time: 2.75 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.0057962, 0.0075394, 0.0057817, 0.0076057, -0.0018095, 0.0017578
1: -0.0010534, 0.0026853, -0.0011816, 0.0028137, -0.0038671, 0.0038669
2: -0.0047316, 0.0229580, -0.0057670, 0.0233100, -0.0280416, 0.0287250
3: -0.0045225, -0.0020900, -0.0045427, -0.0019975, -0.0025250, 0.0024527
4: -0.0006060, 0.0111960, -0.0007041, 0.0116447, -0.0122507, 0.0119001
5: -0.0021911, -0.0001326, -0.0022580, -0.0000360, -0.0021550, 0.0021254
6: 0.9902643, 0.9942716, 0.9900231, 0.9943945, -0.0041302, 0.0042486
7: -0.0146296, 0.0068839, -0.0148486, 0.0076961, -0.0223257, 0.0217325
8: -0.0057620, 0.0031450, -0.0064293, 0.0033995, -0.0091615, 0.0095743
9: -0.0136062, 0.0000459, -0.0141140, 0.0002380, -0.0138442, 0.0141599

Time for backsubstitution: 2.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038211, upper bound: 0.0039132
time: 2.75 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038559, upper bound: 0.0039132
time: 3.17 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.0058420, 0.0074122, 0.0057964, 0.0075260, -0.0016840, 0.0016157
1: -0.0006486, 0.0024388, -0.0010511, 0.0026592, -0.0033079, 0.0034899
2: -0.0027430, 0.0218458, -0.0045213, 0.0229517, -0.0256947, 0.0263671
3: -0.0044586, -0.0022676, -0.0045221, -0.0021088, -0.0023498, 0.0022545
4: -0.0002960, 0.0103343, -0.0006042, 0.0111049, -0.0114009, 0.0109385
5: -0.0020624, -0.0004378, -0.0021775, -0.0001344, -0.0019281, 0.0017397
6: 0.9910264, 0.9940357, 0.9902686, 0.9942467, -0.0032203, 0.0037671
7: -0.0139376, 0.0053240, -0.0146257, 0.0067189, -0.0206565, 0.0199497
8: -0.0036539, 0.0026563, -0.0057501, 0.0030933, -0.0067473, 0.0084064
9: -0.0126308, -0.0005612, -0.0135030, 0.0000424, -0.0126732, 0.0129418

Time for backsubstitution: 2.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037473, upper bound: 0.0036736
time: 3.19 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037510, upper bound: 0.0036460
time: 3.09 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.0058095, 0.0074490, 0.0057964, 0.0075260, -0.0017165, 0.0016525
1: -0.0009356, 0.0025101, -0.0010511, 0.0026592, -0.0035948, 0.0035612
2: -0.0033183, 0.0226341, -0.0045213, 0.0229517, -0.0262700, 0.0271554
3: -0.0045039, -0.0022162, -0.0045221, -0.0021088, -0.0023951, 0.0023059
4: -0.0005157, 0.0105836, -0.0006042, 0.0111049, -0.0116206, 0.0111878
5: -0.0020996, -0.0002215, -0.0021775, -0.0001344, -0.0019653, 0.0019560
6: 0.9904863, 0.9941039, 0.9902686, 0.9942467, -0.0037605, 0.0038353
7: -0.0144281, 0.0057752, -0.0146257, 0.0067189, -0.0211470, 0.0204009
8: -0.0051481, 0.0027977, -0.0057501, 0.0030933, -0.0082415, 0.0085478
9: -0.0129129, -0.0001309, -0.0135030, 0.0000424, -0.0129554, 0.0133721

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037473, upper bound: 0.0037068
time: 2.57 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037510, upper bound: 0.0036697
time: 2.79 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.0058420, 0.0074122, 0.0057534, 0.0076143, -0.0017724, 0.0016588
1: -0.0006486, 0.0024388, -0.0014316, 0.0028304, -0.0034790, 0.0038704
2: -0.0027430, 0.0218458, -0.0059015, 0.0239970, -0.0267400, 0.0277474
3: -0.0044586, -0.0022676, -0.0045822, -0.0019855, -0.0024731, 0.0023146
4: -0.0002960, 0.0103343, -0.0008956, 0.0117030, -0.0119990, 0.0112299
5: -0.0020624, -0.0004378, -0.0022667, 0.0001525, -0.0022149, 0.0018289
6: 0.9910264, 0.9940357, 0.9895521, 0.9944105, -0.0033841, 0.0044836
7: -0.0139376, 0.0053240, -0.0152761, 0.0078016, -0.0217392, 0.0206000
8: -0.0036539, 0.0026563, -0.0077314, 0.0034325, -0.0070865, 0.0103877
9: -0.0126308, -0.0005612, -0.0141800, 0.0006130, -0.0132438, 0.0136188

Time for backsubstitution: 2.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037473, upper bound: 0.0036721
time: 2.39 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037473, upper bound: 0.0036460
time: 3.32 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.0058095, 0.0074490, 0.0057534, 0.0076143, -0.0018048, 0.0016956
1: -0.0009356, 0.0025101, -0.0014316, 0.0028304, -0.0037659, 0.0039417
2: -0.0033183, 0.0226341, -0.0059015, 0.0239970, -0.0273153, 0.0285357
3: -0.0045039, -0.0022162, -0.0045822, -0.0019855, -0.0025184, 0.0023660
4: -0.0005157, 0.0105836, -0.0008956, 0.0117030, -0.0122187, 0.0114792
5: -0.0020996, -0.0002215, -0.0022667, 0.0001525, -0.0022521, 0.0020452
6: 0.9904863, 0.9941039, 0.9895521, 0.9944105, -0.0039242, 0.0045518
7: -0.0144281, 0.0057752, -0.0152761, 0.0078016, -0.0222297, 0.0210513
8: -0.0051481, 0.0027977, -0.0077314, 0.0034325, -0.0085807, 0.0105290
9: -0.0129129, -0.0001309, -0.0141800, 0.0006130, -0.0135260, 0.0140491

Time for backsubstitution: 2.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037473, upper bound: 0.0036987
time: 3.06 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037473, upper bound: 0.0036690
time: 3.22 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: 0.0058251, 0.0075060, 0.0057870, 0.0075606, -0.0017355, 0.0017190
1: -0.0007975, 0.0026204, -0.0011348, 0.0027262, -0.0035237, 0.0037552
2: -0.0042082, 0.0222547, -0.0050614, 0.0231816, -0.0273898, 0.0273161
3: -0.0044821, -0.0021367, -0.0045353, -0.0020605, -0.0024215, 0.0023986
4: -0.0004099, 0.0109692, -0.0006683, 0.0113389, -0.0117489, 0.0116375
5: -0.0021572, -0.0003256, -0.0022124, -0.0000713, -0.0020859, 0.0018868
6: 0.9907463, 0.9942095, 0.9901111, 0.9943108, -0.0035645, 0.0040985
7: -0.0141920, 0.0064733, -0.0147687, 0.0071426, -0.0213346, 0.0212421
8: -0.0044290, 0.0030164, -0.0061858, 0.0032261, -0.0076550, 0.0092022
9: -0.0133495, -0.0003381, -0.0137679, 0.0001679, -0.0135174, 0.0134299

Time for backsubstitution: 2.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038212, upper bound: 0.0039018
time: 3.11 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038212, upper bound: 0.0039018
time: 3.03 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.0057962, 0.0075394, 0.0057870, 0.0075606, -0.0017644, 0.0017525
1: -0.0010534, 0.0026853, -0.0011348, 0.0027262, -0.0037796, 0.0038201
2: -0.0047316, 0.0229580, -0.0050614, 0.0231816, -0.0279132, 0.0280194
3: -0.0045225, -0.0020900, -0.0045353, -0.0020605, -0.0024619, 0.0024453
4: -0.0006060, 0.0111960, -0.0006683, 0.0113389, -0.0119449, 0.0118643
5: -0.0021911, -0.0001326, -0.0022124, -0.0000713, -0.0021198, 0.0020798
6: 0.9902643, 0.9942716, 0.9901111, 0.9943108, -0.0040465, 0.0041606
7: -0.0146296, 0.0068839, -0.0147687, 0.0071426, -0.0217722, 0.0216526
8: -0.0057620, 0.0031450, -0.0061858, 0.0032261, -0.0089881, 0.0093308
9: -0.0136062, 0.0000459, -0.0137679, 0.0001679, -0.0137741, 0.0138138

Time for backsubstitution: 2.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038212, upper bound: 0.0039237
time: 3.17 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038212, upper bound: 0.0039237
time: 2.32 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.0058251, 0.0075060, 0.0057436, 0.0076481, -0.0018230, 0.0017623
1: -0.0007975, 0.0026204, -0.0015177, 0.0028958, -0.0036933, 0.0041381
2: -0.0042082, 0.0222547, -0.0064296, 0.0242336, -0.0284418, 0.0286843
3: -0.0044821, -0.0021367, -0.0045957, -0.0019383, -0.0025437, 0.0024590
4: -0.0004099, 0.0109692, -0.0009616, 0.0119318, -0.0123418, 0.0119308
5: -0.0021572, -0.0003256, -0.0023009, 0.0002174, -0.0023746, 0.0019753
6: 0.9907463, 0.9942095, 0.9893900, 0.9944730, -0.0037268, 0.0048195
7: -0.0141920, 0.0064733, -0.0154233, 0.0082158, -0.0224078, 0.0218966
8: -0.0044290, 0.0030164, -0.0081798, 0.0035623, -0.0079912, 0.0111962
9: -0.0133495, -0.0003381, -0.0144390, 0.0007422, -0.0140916, 0.0141009

Time for backsubstitution: 2.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038211, upper bound: 0.0038975
time: 3.23 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038559, upper bound: 0.0038975
time: 2.75 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.0057962, 0.0075394, 0.0057436, 0.0076481, -0.0018519, 0.0017958
1: -0.0010534, 0.0026853, -0.0015177, 0.0028958, -0.0039493, 0.0042030
2: -0.0047316, 0.0229580, -0.0064296, 0.0242336, -0.0289652, 0.0293876
3: -0.0045225, -0.0020900, -0.0045957, -0.0019383, -0.0025841, 0.0025058
4: -0.0006060, 0.0111960, -0.0009616, 0.0119318, -0.0125378, 0.0121576
5: -0.0021911, -0.0001326, -0.0023009, 0.0002174, -0.0024084, 0.0021683
6: 0.9902643, 0.9942716, 0.9893900, 0.9944730, -0.0042087, 0.0048816
7: -0.0146296, 0.0068839, -0.0154233, 0.0082158, -0.0228454, 0.0223071
8: -0.0057620, 0.0031450, -0.0081798, 0.0035623, -0.0093243, 0.0113248
9: -0.0136062, 0.0000459, -0.0144390, 0.0007422, -0.0143483, 0.0144849

Time for backsubstitution: 2.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038211, upper bound: 0.0039132
time: 2.80 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038559, upper bound: 0.0039132
time: 2.65 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.0058095, 0.0075271, 0.0058074, 0.0074558, -0.0016463, 0.0017197
1: -0.0009354, 0.0026613, -0.0009543, 0.0025234, -0.0034587, 0.0036156
2: -0.0045378, 0.0226337, -0.0034252, 0.0226858, -0.0272236, 0.0260589
3: -0.0045038, -0.0021073, -0.0045068, -0.0022067, -0.0022972, 0.0023995
4: -0.0005156, 0.0111121, -0.0005301, 0.0106299, -0.0111455, 0.0116422
5: -0.0021785, -0.0002216, -0.0021066, -0.0002073, -0.0019712, 0.0018849
6: 0.9904866, 0.9942486, 0.9904508, 0.9941167, -0.0036300, 0.0037978
7: -0.0144278, 0.0067319, -0.0144602, 0.0058591, -0.0202869, 0.0211921
8: -0.0051473, 0.0030974, -0.0052460, 0.0028240, -0.0079712, 0.0083434
9: -0.0135111, -0.0001312, -0.0129654, -0.0001027, -0.0134084, 0.0128342

Time for backsubstitution: 2.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 96

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036742, upper bound: 0.0037857
time: 2.79 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036742, upper bound: 0.0037857
time: 2.91 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.0057448, 0.0075330, 0.0058154, 0.0074431, -0.0016982, 0.0017176
1: -0.0015072, 0.0026729, -0.0008837, 0.0024986, -0.0040059, 0.0035566
2: -0.0046310, 0.0242049, -0.0032257, 0.0224917, -0.0271227, 0.0274306
3: -0.0045941, -0.0020989, -0.0044957, -0.0022245, -0.0023696, 0.0023967
4: -0.0009535, 0.0111524, -0.0004760, 0.0105435, -0.0114970, 0.0116284
5: -0.0021846, 0.0002095, -0.0020937, -0.0002606, -0.0019240, 0.0023032
6: 0.9894097, 0.9942597, 0.9905838, 0.9940930, -0.0046833, 0.0036759
7: -0.0154054, 0.0068050, -0.0143395, 0.0057026, -0.0211080, 0.0211445
8: -0.0081254, 0.0031203, -0.0048782, 0.0027749, -0.0109003, 0.0079985
9: -0.0135568, 0.0007265, -0.0128675, -0.0002087, -0.0133482, 0.0135940

Time for backsubstitution: 2.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 214

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 96

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036469, upper bound: 0.0038012
time: 2.70 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036469, upper bound: 0.0038012
time: 3.05 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.0058054, 0.0074950, 0.0057926, 0.0075508, -0.0017454, 0.0017024
1: -0.0009715, 0.0025991, -0.0010850, 0.0027074, -0.0036789, 0.0036841
2: -0.0040365, 0.0227329, -0.0049096, 0.0230447, -0.0270812, 0.0276425
3: -0.0045095, -0.0021520, -0.0045274, -0.0020741, -0.0024355, 0.0023754
4: -0.0005432, 0.0108948, -0.0006302, 0.0112732, -0.0118164, 0.0115250
5: -0.0021461, -0.0001944, -0.0022026, -0.0001088, -0.0020373, 0.0020082
6: 0.9904186, 0.9941892, 0.9902049, 0.9942927, -0.0038742, 0.0039843
7: -0.0144896, 0.0063386, -0.0146836, 0.0070235, -0.0215131, 0.0210222
8: -0.0053353, 0.0029742, -0.0059264, 0.0031888, -0.0085241, 0.0089006
9: -0.0132652, -0.0000770, -0.0136935, 0.0000932, -0.0133584, 0.0136165

Time for backsubstitution: 2.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 96

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038217, upper bound: 0.0038560
time: 3.07 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038217, upper bound: 0.0038559
time: 6.37 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.0057908, 0.0075845, 0.0057926, 0.0075508, -0.0017601, 0.0017919
1: -0.0011013, 0.0027726, -0.0010850, 0.0027074, -0.0038087, 0.0038576
2: -0.0054356, 0.0230895, -0.0049096, 0.0230447, -0.0284803, 0.0279991
3: -0.0045300, -0.0020271, -0.0045274, -0.0020741, -0.0024559, 0.0025004
4: -0.0006426, 0.0115011, -0.0006302, 0.0112732, -0.0119158, 0.0121312
5: -0.0022366, -0.0000966, -0.0022026, -0.0001088, -0.0021278, 0.0021060
6: 0.9901741, 0.9943551, 0.9902049, 0.9942927, -0.0041186, 0.0041503
7: -0.0147114, 0.0074361, -0.0146836, 0.0070235, -0.0217349, 0.0221197
8: -0.0060113, 0.0033180, -0.0059264, 0.0031888, -0.0092000, 0.0092444
9: -0.0139515, 0.0001176, -0.0136935, 0.0000932, -0.0140447, 0.0138111

Time for backsubstitution: 2.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 96

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038217, upper bound: 0.0039429
time: 2.82 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038217, upper bound: 0.0039429
time: 2.25 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: 0.0057713, 0.0075697, 0.0058038, 0.0074683, -0.0016970, 0.0017659
1: -0.0012733, 0.0027439, -0.0009862, 0.0025475, -0.0038207, 0.0037301
2: -0.0052039, 0.0235620, -0.0036196, 0.0227733, -0.0279772, 0.0271817
3: -0.0045572, -0.0020478, -0.0045118, -0.0021893, -0.0023679, 0.0024641
4: -0.0007744, 0.0114007, -0.0005545, 0.0107142, -0.0114885, 0.0119552
5: -0.0022216, 0.0000331, -0.0021191, -0.0001833, -0.0020383, 0.0021522
6: 0.9898503, 0.9943277, 0.9903909, 0.9941397, -0.0042894, 0.0039368
7: -0.0150054, 0.0072543, -0.0145147, 0.0060116, -0.0210171, 0.0217690
8: -0.0069069, 0.0032611, -0.0054119, 0.0028717, -0.0097787, 0.0086730
9: -0.0138378, 0.0003756, -0.0130608, -0.0000550, -0.0137829, 0.0134363

Time for backsubstitution: 2.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 214

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 96

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036721, upper bound: 0.0038232
time: 2.63 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036721, upper bound: 0.0038232
time: 2.77 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.0057077, 0.0075763, 0.0058118, 0.0074558, -0.0017481, 0.0017645
1: -0.0018355, 0.0027566, -0.0009157, 0.0025233, -0.0043588, 0.0036723
2: -0.0053066, 0.0251068, -0.0034245, 0.0225796, -0.0278862, 0.0285313
3: -0.0046459, -0.0020386, -0.0045007, -0.0022067, -0.0024392, 0.0024621
4: -0.0012050, 0.0114452, -0.0005005, 0.0106296, -0.0118346, 0.0119457
5: -0.0022283, 0.0004570, -0.0021065, -0.0002365, -0.0019918, 0.0025635
6: 0.9887916, 0.9943398, 0.9905236, 0.9941166, -0.0053250, 0.0038162
7: -0.0159665, 0.0073349, -0.0143942, 0.0058586, -0.0218251, 0.0217291
8: -0.0098349, 0.0032863, -0.0050447, 0.0028238, -0.0126587, 0.0083311
9: -0.0138882, 0.0012188, -0.0129650, -0.0001607, -0.0137275, 0.0141839

Time for backsubstitution: 2.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 214

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 96

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036460, upper bound: 0.0038502
time: 2.39 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036460, upper bound: 0.0038502
time: 2.10 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.0057645, 0.0075392, 0.0057888, 0.0075625, -0.0017980, 0.0017504
1: -0.0013337, 0.0026849, -0.0011183, 0.0027299, -0.0040636, 0.0038032
2: -0.0047281, 0.0237279, -0.0050914, 0.0231362, -0.0278643, 0.0288193
3: -0.0045667, -0.0020903, -0.0045327, -0.0020578, -0.0025089, 0.0024424
4: -0.0008206, 0.0111945, -0.0006557, 0.0113519, -0.0121725, 0.0118502
5: -0.0021908, 0.0000786, -0.0022143, -0.0000837, -0.0021071, 0.0022930
6: 0.9897366, 0.9942713, 0.9901422, 0.9943143, -0.0045778, 0.0041291
7: -0.0151087, 0.0068811, -0.0147405, 0.0071661, -0.0222747, 0.0216216
8: -0.0072214, 0.0031442, -0.0060998, 0.0032334, -0.0104548, 0.0092440
9: -0.0136044, 0.0004662, -0.0137826, 0.0001431, -0.0137476, 0.0142488

Time for backsubstitution: 2.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 96

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038182, upper bound: 0.0039111
time: 2.14 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038182, upper bound: 0.0039111
time: 2.76 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.0057530, 0.0076257, 0.0057888, 0.0075625, -0.0018095, 0.0018369
1: -0.0014353, 0.0028524, -0.0011183, 0.0027299, -0.0041652, 0.0039707
2: -0.0060796, 0.0240072, -0.0050914, 0.0231362, -0.0292158, 0.0290986
3: -0.0045827, -0.0019696, -0.0045327, -0.0020578, -0.0025249, 0.0025631
4: -0.0008985, 0.0117802, -0.0006557, 0.0113519, -0.0122504, 0.0124358
5: -0.0022783, 0.0001553, -0.0022143, -0.0000837, -0.0021945, 0.0023696
6: 0.9895453, 0.9944316, 0.9901422, 0.9943143, -0.0047690, 0.0042894
7: -0.0152824, 0.0079413, -0.0147405, 0.0071661, -0.0224485, 0.0226818
8: -0.0077507, 0.0034763, -0.0060998, 0.0032334, -0.0109842, 0.0095761
9: -0.0142673, 0.0006186, -0.0137826, 0.0001431, -0.0144105, 0.0144012

Time for backsubstitution: 2.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 96

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038182, upper bound: 0.0039888
time: 3.00 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038182, upper bound: 0.0039888
time: 3.15 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.0058095, 0.0075271, 0.0057620, 0.0075416, -0.0017321, 0.0017650
1: -0.0009354, 0.0026613, -0.0013554, 0.0026895, -0.0036249, 0.0040166
2: -0.0045378, 0.0226337, -0.0047656, 0.0237875, -0.0283254, 0.0273992
3: -0.0045038, -0.0021073, -0.0045701, -0.0020869, -0.0024169, 0.0024628
4: -0.0005156, 0.0111121, -0.0008372, 0.0112107, -0.0117263, 0.0119493
5: -0.0021785, -0.0002216, -0.0021933, 0.0000950, -0.0022735, 0.0019716
6: 0.9904866, 0.9942486, 0.9896958, 0.9942757, -0.0037891, 0.0045528
7: -0.0144278, 0.0067319, -0.0151457, 0.0069105, -0.0213383, 0.0218776
8: -0.0051473, 0.0030974, -0.0073344, 0.0031534, -0.0083006, 0.0104317
9: -0.0135111, -0.0001312, -0.0136228, 0.0004987, -0.0140098, 0.0134916

Time for backsubstitution: 2.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 96

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036742, upper bound: 0.0037857
time: 3.10 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036742, upper bound: 0.0037857
time: 2.35 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.0057448, 0.0075330, 0.0057698, 0.0075295, -0.0017847, 0.0017632
1: -0.0015072, 0.0026729, -0.0012864, 0.0026661, -0.0041734, 0.0039593
2: -0.0046310, 0.0242049, -0.0045767, 0.0235982, -0.0282292, 0.0287815
3: -0.0045941, -0.0020989, -0.0045592, -0.0021038, -0.0024903, 0.0024603
4: -0.0009535, 0.0111524, -0.0007844, 0.0111289, -0.0120824, 0.0119369
5: -0.0021846, 0.0002095, -0.0021810, 0.0000430, -0.0022276, 0.0023906
6: 0.9894097, 0.9942597, 0.9898255, 0.9942533, -0.0048435, 0.0044342
7: -0.0154054, 0.0068050, -0.0150280, 0.0067624, -0.0221677, 0.0218330
8: -0.0081254, 0.0031203, -0.0069755, 0.0031069, -0.0112323, 0.0100958
9: -0.0135568, 0.0007265, -0.0135302, 0.0003953, -0.0139522, 0.0142567

Time for backsubstitution: 2.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 214

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 96

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036489, upper bound: 0.0038012
time: 2.66 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036489, upper bound: 0.0038012
time: 3.09 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.0058054, 0.0074950, 0.0057491, 0.0076331, -0.0018277, 0.0017459
1: -0.0009715, 0.0025991, -0.0014699, 0.0028668, -0.0038383, 0.0040691
2: -0.0040365, 0.0227329, -0.0061952, 0.0241023, -0.0281388, 0.0289281
3: -0.0045095, -0.0021520, -0.0045882, -0.0019592, -0.0025503, 0.0024362
4: -0.0005432, 0.0108948, -0.0009250, 0.0118303, -0.0123735, 0.0118198
5: -0.0021461, -0.0001944, -0.0022857, 0.0001814, -0.0023275, 0.0020913
6: 0.9904186, 0.9941892, 0.9894801, 0.9944453, -0.0040267, 0.0047091
7: -0.0144896, 0.0063386, -0.0153416, 0.0080320, -0.0225215, 0.0216802
8: -0.0053353, 0.0029742, -0.0079309, 0.0035047, -0.0088400, 0.0109051
9: -0.0132652, -0.0000770, -0.0143240, 0.0006705, -0.0139357, 0.0142470

Time for backsubstitution: 2.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 96

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038217, upper bound: 0.0038559
time: 2.76 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038217, upper bound: 0.0038559
time: 3.47 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.0057908, 0.0075845, 0.0057491, 0.0076331, -0.0018424, 0.0018354
1: -0.0011013, 0.0027726, -0.0014699, 0.0028668, -0.0039681, 0.0042425
2: -0.0054356, 0.0230895, -0.0061952, 0.0241023, -0.0295379, 0.0292847
3: -0.0045300, -0.0020271, -0.0045882, -0.0019592, -0.0025708, 0.0025611
4: -0.0006426, 0.0115011, -0.0009250, 0.0118303, -0.0124729, 0.0124260
5: -0.0022366, -0.0000966, -0.0022857, 0.0001814, -0.0024180, 0.0021892
6: 0.9901741, 0.9943551, 0.9894801, 0.9944453, -0.0042711, 0.0048750
7: -0.0147114, 0.0074361, -0.0153416, 0.0080320, -0.0227434, 0.0227777
8: -0.0060113, 0.0033180, -0.0079309, 0.0035047, -0.0095159, 0.0112489
9: -0.0139515, 0.0001176, -0.0143240, 0.0006705, -0.0146220, 0.0144417

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 96

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038217, upper bound: 0.0039429
time: 2.39 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038217, upper bound: 0.0039429
time: 2.13 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: 0.0057713, 0.0075697, 0.0057587, 0.0075560, -0.0017847, 0.0018110
1: -0.0012733, 0.0027439, -0.0013850, 0.0027174, -0.0039907, 0.0041289
2: -0.0052039, 0.0235620, -0.0049905, 0.0238691, -0.0290730, 0.0285526
3: -0.0045572, -0.0020478, -0.0045748, -0.0020668, -0.0024903, 0.0025270
4: -0.0007744, 0.0114007, -0.0008600, 0.0113082, -0.0120826, 0.0122606
5: -0.0022216, 0.0000331, -0.0022078, 0.0001174, -0.0023390, 0.0022409
6: 0.9898503, 0.9943277, 0.9896399, 0.9943023, -0.0044520, 0.0046878
7: -0.0150054, 0.0072543, -0.0151965, 0.0070870, -0.0220924, 0.0224508
8: -0.0069069, 0.0032611, -0.0074890, 0.0032086, -0.0101156, 0.0107501
9: -0.0138378, 0.0003756, -0.0137332, 0.0005432, -0.0143810, 0.0141088

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 96

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036722, upper bound: 0.0038232
time: 2.97 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036722, upper bound: 0.0038232
time: 2.12 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.0057077, 0.0075763, 0.0057664, 0.0075438, -0.0018361, 0.0018098
1: -0.0018355, 0.0027566, -0.0013163, 0.0026938, -0.0045293, 0.0040729
2: -0.0053066, 0.0251068, -0.0047999, 0.0236803, -0.0289869, 0.0299067
3: -0.0046459, -0.0020386, -0.0045640, -0.0020839, -0.0025620, 0.0025253
4: -0.0012050, 0.0114452, -0.0008073, 0.0112256, -0.0124306, 0.0122525
5: -0.0022283, 0.0004570, -0.0021955, 0.0000656, -0.0022938, 0.0026525
6: 0.9887916, 0.9943398, 0.9897693, 0.9942797, -0.0054881, 0.0045705
7: -0.0159665, 0.0073349, -0.0150790, 0.0069375, -0.0229040, 0.0224139
8: -0.0098349, 0.0032863, -0.0071310, 0.0031618, -0.0129967, 0.0104174
9: -0.0138882, 0.0012188, -0.0136397, 0.0004401, -0.0143283, 0.0148585

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 214

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 96

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036486, upper bound: 0.0038502
time: 2.33 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036486, upper bound: 0.0038501
time: 2.37 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.0057645, 0.0075392, 0.0057455, 0.0076467, -0.0018822, 0.0017937
1: -0.0013337, 0.0026849, -0.0015011, 0.0028930, -0.0042266, 0.0041860
2: -0.0047281, 0.0237279, -0.0064065, 0.0241880, -0.0289161, 0.0301344
3: -0.0045667, -0.0020903, -0.0045931, -0.0019404, -0.0026263, 0.0025029
4: -0.0008206, 0.0111945, -0.0009488, 0.0119218, -0.0127424, 0.0121434
5: -0.0021908, 0.0000786, -0.0022994, 0.0002049, -0.0023957, 0.0023780
6: 0.9897366, 0.9942713, 0.9894214, 0.9944704, -0.0047339, 0.0048499
7: -0.0151087, 0.0068811, -0.0153949, 0.0081977, -0.0233063, 0.0222760
8: -0.0072214, 0.0031442, -0.0080934, 0.0035566, -0.0107780, 0.0112375
9: -0.0136044, 0.0004662, -0.0144277, 0.0007173, -0.0143217, 0.0148938

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 96

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038182, upper bound: 0.0039111
time: 2.07 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038182, upper bound: 0.0039111
time: 2.60 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.0057530, 0.0076257, 0.0057455, 0.0076467, -0.0018937, 0.0018802
1: -0.0014353, 0.0028524, -0.0015011, 0.0028930, -0.0043283, 0.0043535
2: -0.0060796, 0.0240072, -0.0064065, 0.0241880, -0.0302676, 0.0304137
3: -0.0045827, -0.0019696, -0.0045931, -0.0019404, -0.0026424, 0.0026236
4: -0.0008985, 0.0117802, -0.0009488, 0.0119218, -0.0128203, 0.0127290
5: -0.0022783, 0.0001553, -0.0022994, 0.0002049, -0.0024831, 0.0024547
6: 0.9895453, 0.9944316, 0.9894214, 0.9944704, -0.0049251, 0.0050102
7: -0.0152824, 0.0079413, -0.0153949, 0.0081977, -0.0234801, 0.0233362
8: -0.0077507, 0.0034763, -0.0080934, 0.0035566, -0.0113073, 0.0115697
9: -0.0142673, 0.0006186, -0.0144277, 0.0007173, -0.0149846, 0.0150463

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 96

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038182, upper bound: 0.0039888
time: 2.28 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038182, upper bound: 0.0039888
time: 2.69 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 6.78 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 6.78
Output dim: 6, lower bound: -0.0034326, upper bound: 0.0033983
NS_A1_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 6.78
Output dim: 6, lower bound: -0.0034366, upper bound: 0.0032868
NS_A1_B1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 6.78
Output dim: 6, lower bound: -0.0033683, upper bound: 0.0033840
NS_A1_B1_A1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 6.78
Output dim: 6, lower bound: -0.0033633, upper bound: 0.0032554
NS_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 6.78
Output dim: 6, lower bound: -0.0034928, upper bound: 0.0033983
NS_A1_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 6.78
Output dim: 6, lower bound: -0.0034999, upper bound: 0.0032868
NS_A1_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 6.78
Output dim: 6, lower bound: -0.0034162, upper bound: 0.0033840
NS_A1_B1_A1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 6.78
Output dim: 6, lower bound: -0.0034137, upper bound: 0.0032554
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.78
Output dim: 6, lower bound: -0.0038212, upper bound: 0.0039072
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.78
Output dim: 6, lower bound: -0.0038212, upper bound: 0.0039072
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.78
Output dim: 6, lower bound: -0.0038212, upper bound: 0.0039237
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.78
Output dim: 6, lower bound: -0.0038212, upper bound: 0.0039237
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.78
Output dim: 6, lower bound: -0.0038211, upper bound: 0.0038998
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.78
Output dim: 6, lower bound: -0.0038559, upper bound: 0.0038998
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.78
Output dim: 6, lower bound: -0.0038211, upper bound: 0.0039132
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.78
Output dim: 6, lower bound: -0.0038559, upper bound: 0.0039132
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.78
Output dim: 6, lower bound: -0.0037473, upper bound: 0.0036736
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.78
Output dim: 6, lower bound: -0.0037510, upper bound: 0.0036460
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.78
Output dim: 6, lower bound: -0.0037473, upper bound: 0.0037068
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.78
Output dim: 6, lower bound: -0.0037510, upper bound: 0.0036697
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.78
Output dim: 6, lower bound: -0.0037473, upper bound: 0.0036721
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.78
Output dim: 6, lower bound: -0.0037473, upper bound: 0.0036460
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.78
Output dim: 6, lower bound: -0.0037473, upper bound: 0.0036987
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.78
Output dim: 6, lower bound: -0.0037473, upper bound: 0.0036690
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.78
Output dim: 6, lower bound: -0.0038212, upper bound: 0.0039018
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.78
Output dim: 6, lower bound: -0.0038212, upper bound: 0.0039018
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.78
Output dim: 6, lower bound: -0.0038212, upper bound: 0.0039237
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.78
Output dim: 6, lower bound: -0.0038212, upper bound: 0.0039237
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.78
Output dim: 6, lower bound: -0.0038211, upper bound: 0.0038975
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.78
Output dim: 6, lower bound: -0.0038559, upper bound: 0.0038975
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.78
Output dim: 6, lower bound: -0.0038211, upper bound: 0.0039132
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.78
Output dim: 6, lower bound: -0.0038559, upper bound: 0.0039132
NS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.78
Output dim: 6, lower bound: -0.0036742, upper bound: 0.0037857
NS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.78
Output dim: 6, lower bound: -0.0036742, upper bound: 0.0037857
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.78
Output dim: 6, lower bound: -0.0036469, upper bound: 0.0038012
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.78
Output dim: 6, lower bound: -0.0036469, upper bound: 0.0038012
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.78
Output dim: 6, lower bound: -0.0038217, upper bound: 0.0038560
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.78
Output dim: 6, lower bound: -0.0038217, upper bound: 0.0038559
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.78
Output dim: 6, lower bound: -0.0038217, upper bound: 0.0039429
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.78
Output dim: 6, lower bound: -0.0038217, upper bound: 0.0039429
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.78
Output dim: 6, lower bound: -0.0036721, upper bound: 0.0038232
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.78
Output dim: 6, lower bound: -0.0036721, upper bound: 0.0038232
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.78
Output dim: 6, lower bound: -0.0036460, upper bound: 0.0038502
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.78
Output dim: 6, lower bound: -0.0036460, upper bound: 0.0038502
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.78
Output dim: 6, lower bound: -0.0038182, upper bound: 0.0039111
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.78
Output dim: 6, lower bound: -0.0038182, upper bound: 0.0039111
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.78
Output dim: 6, lower bound: -0.0038182, upper bound: 0.0039888
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.78
Output dim: 6, lower bound: -0.0038182, upper bound: 0.0039888
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.78
Output dim: 6, lower bound: -0.0036742, upper bound: 0.0037857
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.78
Output dim: 6, lower bound: -0.0036742, upper bound: 0.0037857
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.78
Output dim: 6, lower bound: -0.0036489, upper bound: 0.0038012
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.78
Output dim: 6, lower bound: -0.0036489, upper bound: 0.0038012
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.78
Output dim: 6, lower bound: -0.0038217, upper bound: 0.0038559
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.78
Output dim: 6, lower bound: -0.0038217, upper bound: 0.0038559
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.78
Output dim: 6, lower bound: -0.0038217, upper bound: 0.0039429
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.78
Output dim: 6, lower bound: -0.0038217, upper bound: 0.0039429
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.78
Output dim: 6, lower bound: -0.0036722, upper bound: 0.0038232
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.78
Output dim: 6, lower bound: -0.0036722, upper bound: 0.0038232
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.78
Output dim: 6, lower bound: -0.0036486, upper bound: 0.0038502
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.78
Output dim: 6, lower bound: -0.0036486, upper bound: 0.0038501
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.78
Output dim: 6, lower bound: -0.0038182, upper bound: 0.0039111
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.78
Output dim: 6, lower bound: -0.0038182, upper bound: 0.0039111
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.78
Output dim: 6, lower bound: -0.0038182, upper bound: 0.0039888
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.78
Output dim: 6, lower bound: -0.0038182, upper bound: 0.0039888

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.0058251, 0.0075060, 0.0058420, 0.0074122, -0.0015870, 0.0016640
1: -0.0007975, 0.0026204, -0.0006486, 0.0024388, -0.0032362, 0.0032691
2: -0.0042082, 0.0222547, -0.0027430, 0.0218458, -0.0260541, 0.0249977
3: -0.0044821, -0.0021367, -0.0044586, -0.0022676, -0.0022145, 0.0023219
4: -0.0004099, 0.0109692, -0.0002960, 0.0103343, -0.0107442, 0.0112652
5: -0.0021572, -0.0003256, -0.0020624, -0.0004378, -0.0017194, 0.0017368
6: 0.9907463, 0.9942095, 0.9910264, 0.9940357, -0.0032895, 0.0031831
7: -0.0141920, 0.0064733, -0.0139376, 0.0053240, -0.0195160, 0.0204110
8: -0.0044290, 0.0030164, -0.0036539, 0.0026563, -0.0070853, 0.0066703
9: -0.0133495, -0.0003381, -0.0126308, -0.0005612, -0.0127882, 0.0122927

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 214

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036776, upper bound: 0.0037474
time: 2.81 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036469, upper bound: 0.0037513
time: 2.76 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.0058251, 0.0075060, 0.0058251, 0.0075060, -0.0016808, 0.0016808
1: -0.0007975, 0.0026204, -0.0007975, 0.0026204, -0.0034179, 0.0034179
2: -0.0042082, 0.0222547, -0.0042082, 0.0222547, -0.0264629, 0.0264629
3: -0.0044821, -0.0021367, -0.0044821, -0.0021367, -0.0023453, 0.0023453
4: -0.0004099, 0.0109692, -0.0004099, 0.0109692, -0.0113792, 0.0113792
5: -0.0021572, -0.0003256, -0.0021572, -0.0003256, -0.0018316, 0.0018316
6: 0.9907463, 0.9942095, 0.9907463, 0.9942095, -0.0034633, 0.0034633
7: -0.0141920, 0.0064733, -0.0141920, 0.0064733, -0.0206654, 0.0206654
8: -0.0044290, 0.0030164, -0.0044290, 0.0030164, -0.0074453, 0.0074453
9: -0.0133495, -0.0003381, -0.0133495, -0.0003381, -0.0130114, 0.0130114

Time for backsubstitution: 2.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 214

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036776, upper bound: 0.0037668
time: 3.01 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036469, upper bound: 0.0037718
time: 2.92 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.0057962, 0.0075394, 0.0058420, 0.0074122, -0.0016160, 0.0016975
1: -0.0010534, 0.0026853, -0.0006486, 0.0024388, -0.0034922, 0.0033340
2: -0.0047316, 0.0229580, -0.0027430, 0.0218458, -0.0265774, 0.0257010
3: -0.0045225, -0.0020900, -0.0044586, -0.0022676, -0.0022549, 0.0023686
4: -0.0006060, 0.0111960, -0.0002960, 0.0103343, -0.0109403, 0.0114920
5: -0.0021911, -0.0001326, -0.0020624, -0.0004378, -0.0017533, 0.0019298
6: 0.9902643, 0.9942716, 0.9910264, 0.9940357, -0.0037714, 0.0032452
7: -0.0146296, 0.0068839, -0.0139376, 0.0053240, -0.0199536, 0.0208215
8: -0.0057620, 0.0031450, -0.0036539, 0.0026563, -0.0084183, 0.0067990
9: -0.0136062, 0.0000459, -0.0126308, -0.0005612, -0.0130449, 0.0126766

Time for backsubstitution: 2.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 214

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036736, upper bound: 0.0037614
time: 2.99 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036460, upper bound: 0.0037690
time: 2.84 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.0057962, 0.0075394, 0.0058251, 0.0075060, -0.0017098, 0.0017143
1: -0.0010534, 0.0026853, -0.0007975, 0.0026204, -0.0036739, 0.0034828
2: -0.0047316, 0.0229580, -0.0042082, 0.0222547, -0.0269863, 0.0271662
3: -0.0045225, -0.0020900, -0.0044821, -0.0021367, -0.0023857, 0.0023921
4: -0.0006060, 0.0111960, -0.0004099, 0.0109692, -0.0115752, 0.0116060
5: -0.0021911, -0.0001326, -0.0021572, -0.0003256, -0.0018654, 0.0020246
6: 0.9902643, 0.9942716, 0.9907463, 0.9942095, -0.0039452, 0.0035254
7: -0.0146296, 0.0068839, -0.0141920, 0.0064733, -0.0211030, 0.0210759
8: -0.0057620, 0.0031450, -0.0044290, 0.0030164, -0.0087784, 0.0075740
9: -0.0136062, 0.0000459, -0.0133495, -0.0003381, -0.0132681, 0.0133953

Time for backsubstitution: 2.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 214

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036736, upper bound: 0.0037817
time: 2.75 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036460, upper bound: 0.0037906
time: 2.56 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.0058251, 0.0075060, 0.0058054, 0.0074950, -0.0016698, 0.0017005
1: -0.0007975, 0.0026204, -0.0009715, 0.0025991, -0.0033966, 0.0035919
2: -0.0042082, 0.0222547, -0.0040365, 0.0227329, -0.0269411, 0.0262912
3: -0.0044821, -0.0021367, -0.0045095, -0.0021520, -0.0023300, 0.0023728
4: -0.0004099, 0.0109692, -0.0005432, 0.0108948, -0.0113047, 0.0115125
5: -0.0021572, -0.0003256, -0.0021461, -0.0001944, -0.0019628, 0.0018205
6: 0.9907463, 0.9942095, 0.9904186, 0.9941892, -0.0034429, 0.0037910
7: -0.0141920, 0.0064733, -0.0144896, 0.0063386, -0.0205306, 0.0209629
8: -0.0044290, 0.0030164, -0.0053353, 0.0029742, -0.0074031, 0.0083517
9: -0.0133495, -0.0003381, -0.0132652, -0.0000770, -0.0132724, 0.0129272

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 214

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037458, upper bound: 0.0037474
time: 2.64 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036469, upper bound: 0.0037513
time: 3.21 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.0058251, 0.0075060, 0.0057908, 0.0075845, -0.0017594, 0.0017152
1: -0.0007975, 0.0026204, -0.0011013, 0.0027726, -0.0035701, 0.0037217
2: -0.0042082, 0.0222547, -0.0054356, 0.0230895, -0.0272977, 0.0276903
3: -0.0044821, -0.0021367, -0.0045300, -0.0020271, -0.0024550, 0.0023933
4: -0.0004099, 0.0109692, -0.0006426, 0.0115011, -0.0119110, 0.0116119
5: -0.0021572, -0.0003256, -0.0022366, -0.0000966, -0.0020607, 0.0019110
6: 0.9907463, 0.9942095, 0.9901741, 0.9943551, -0.0036089, 0.0040354
7: -0.0141920, 0.0064733, -0.0147114, 0.0074361, -0.0216281, 0.0211848
8: -0.0044290, 0.0030164, -0.0060113, 0.0033180, -0.0077470, 0.0090276
9: -0.0133495, -0.0003381, -0.0139515, 0.0001176, -0.0134671, 0.0136134

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 214

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037458, upper bound: 0.0037668
time: 2.70 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037003, upper bound: 0.0037712
time: 2.45 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.0057962, 0.0075394, 0.0058054, 0.0074950, -0.0016988, 0.0017340
1: -0.0010534, 0.0026853, -0.0009715, 0.0025991, -0.0036526, 0.0036568
2: -0.0047316, 0.0229580, -0.0040365, 0.0227329, -0.0274645, 0.0269945
3: -0.0045225, -0.0020900, -0.0045095, -0.0021520, -0.0023704, 0.0024196
4: -0.0006060, 0.0111960, -0.0005432, 0.0108948, -0.0115008, 0.0117393
5: -0.0021911, -0.0001326, -0.0021461, -0.0001944, -0.0019967, 0.0020135
6: 0.9902643, 0.9942716, 0.9904186, 0.9941892, -0.0039249, 0.0038531
7: -0.0146296, 0.0068839, -0.0144896, 0.0063386, -0.0209682, 0.0213734
8: -0.0057620, 0.0031450, -0.0053353, 0.0029742, -0.0087362, 0.0084803
9: -0.0136062, 0.0000459, -0.0132652, -0.0000770, -0.0135291, 0.0133111

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 214

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036736, upper bound: 0.0037612
time: 2.73 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036925, upper bound: 0.0037679
time: 3.14 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.0057962, 0.0075394, 0.0057908, 0.0075845, -0.0017883, 0.0017487
1: -0.0010534, 0.0026853, -0.0011013, 0.0027726, -0.0038260, 0.0037866
2: -0.0047316, 0.0229580, -0.0054356, 0.0230895, -0.0278211, 0.0283936
3: -0.0045225, -0.0020900, -0.0045300, -0.0020271, -0.0024954, 0.0024400
4: -0.0006060, 0.0111960, -0.0006426, 0.0115011, -0.0121071, 0.0118387
5: -0.0021911, -0.0001326, -0.0022366, -0.0000966, -0.0020945, 0.0021040
6: 0.9902643, 0.9942716, 0.9901741, 0.9943551, -0.0040908, 0.0040975
7: -0.0146296, 0.0068839, -0.0147114, 0.0074361, -0.0220657, 0.0215953
8: -0.0057620, 0.0031450, -0.0060113, 0.0033180, -0.0090800, 0.0091563
9: -0.0136062, 0.0000459, -0.0139515, 0.0001176, -0.0137238, 0.0139973

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 214

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037305, upper bound: 0.0037809
time: 2.56 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036925, upper bound: 0.0037885
time: 2.08 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.0058434, 0.0074089, 0.0058140, 0.0074820, -0.0016386, 0.0015949
1: -0.0006360, 0.0024325, -0.0008958, 0.0025741, -0.0032101, 0.0033283
2: -0.0026927, 0.0218110, -0.0038345, 0.0225249, -0.0252176, 0.0256455
3: -0.0044566, -0.0022721, -0.0044976, -0.0021701, -0.0022865, 0.0022255
4: -0.0002863, 0.0103125, -0.0004853, 0.0108073, -0.0110935, 0.0107978
5: -0.0020592, -0.0004474, -0.0021330, -0.0002515, -0.0018077, 0.0016857
6: 0.9910504, 0.9940298, 0.9905611, 0.9941652, -0.0031149, 0.0034687
7: -0.0139160, 0.0052845, -0.0143602, 0.0061802, -0.0200962, 0.0196447
8: -0.0035880, 0.0026439, -0.0049411, 0.0029245, -0.0065126, 0.0075851
9: -0.0126061, -0.0005802, -0.0131661, -0.0001905, -0.0124156, 0.0125859

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 214

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0035349, upper bound: 0.0032868
time: 2.58 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0033784, upper bound: 0.0032687
time: 2.60 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.0058509, 0.0073957, 0.0057535, 0.0074875, -0.0016366, 0.0014417
1: -0.0005852, 0.0024069, -0.0014306, 0.0025846, -0.0027835, 0.0038375
2: -0.0024858, 0.0216484, -0.0039195, 0.0239943, -0.0264801, 0.0224516
3: -0.0044461, -0.0022905, -0.0045820, -0.0021625, -0.0022836, 0.0020117
4: -0.0002355, 0.0102228, -0.0008949, 0.0108441, -0.0110796, 0.0097602
5: -0.0020458, -0.0004846, -0.0021385, 0.0001517, -0.0021975, 0.0014524
6: 0.9911419, 0.9940053, 0.9895541, 0.9941753, -0.0026637, 0.0044512
7: -0.0138091, 0.0051222, -0.0152744, 0.0062469, -0.0176114, 0.0203966
8: -0.0033379, 0.0025931, -0.0077263, 0.0029454, -0.0055175, 0.0103194
9: -0.0125046, -0.0006671, -0.0132078, 0.0006116, -0.0131162, 0.0110123

Time for backsubstitution: 2.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 214

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0035433, upper bound: 0.0032600
time: 2.52 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0033766, upper bound: 0.0032362
time: 2.49 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.0058109, 0.0074456, 0.0058140, 0.0074820, -0.0016711, 0.0016316
1: -0.0009231, 0.0025036, -0.0008958, 0.0025741, -0.0034972, 0.0033994
2: -0.0032658, 0.0226000, -0.0038345, 0.0225249, -0.0257907, 0.0264345
3: -0.0045019, -0.0022209, -0.0044976, -0.0021701, -0.0023318, 0.0022767
4: -0.0005062, 0.0105608, -0.0004853, 0.0108073, -0.0113134, 0.0110461
5: -0.0020962, -0.0002309, -0.0021330, -0.0002515, -0.0018448, 0.0019022
6: 0.9905096, 0.9940977, 0.9905611, 0.9941652, -0.0036556, 0.0035366
7: -0.0144069, 0.0057341, -0.0143602, 0.0061802, -0.0205870, 0.0200943
8: -0.0050834, 0.0027848, -0.0049411, 0.0029245, -0.0080079, 0.0077259
9: -0.0128872, -0.0001496, -0.0131661, -0.0001905, -0.0126967, 0.0130166

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 214

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0035266, upper bound: 0.0033185
time: 2.34 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0033669, upper bound: 0.0032963
time: 2.41 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.0058188, 0.0074335, 0.0057535, 0.0074875, -0.0016686, 0.0016799
1: -0.0008531, 0.0024800, -0.0014306, 0.0025846, -0.0034377, 0.0039107
2: -0.0030759, 0.0224075, -0.0039195, 0.0239943, -0.0270702, 0.0263270
3: -0.0044908, -0.0022378, -0.0045820, -0.0021625, -0.0023283, 0.0023441
4: -0.0004525, 0.0104785, -0.0008949, 0.0108441, -0.0112966, 0.0113734
5: -0.0020840, -0.0002837, -0.0021385, 0.0001517, -0.0022357, 0.0018549
6: 0.9906415, 0.9940752, 0.9895541, 0.9941753, -0.0035338, 0.0045211
7: -0.0142871, 0.0055851, -0.0152744, 0.0062469, -0.0205340, 0.0208595
8: -0.0047186, 0.0027381, -0.0077263, 0.0029454, -0.0076641, 0.0104644
9: -0.0127940, -0.0002546, -0.0132078, 0.0006116, -0.0134056, 0.0129532

Time for backsubstitution: 2.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 214

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0035266, upper bound: 0.0032838
time: 2.17 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0033633, upper bound: 0.0032554
time: 3.65 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.0058434, 0.0074089, 0.0057713, 0.0075697, -0.0017263, 0.0016376
1: -0.0006360, 0.0024325, -0.0012733, 0.0027439, -0.0033799, 0.0037058
2: -0.0026927, 0.0218110, -0.0052039, 0.0235620, -0.0262547, 0.0270149
3: -0.0044566, -0.0022721, -0.0045572, -0.0020478, -0.0024088, 0.0022851
4: -0.0002863, 0.0103125, -0.0007744, 0.0114007, -0.0116869, 0.0110869
5: -0.0020592, -0.0004474, -0.0022216, 0.0000331, -0.0020923, 0.0017743
6: 0.9910504, 0.9940298, 0.9898503, 0.9943277, -0.0032773, 0.0041795
7: -0.0139160, 0.0052845, -0.0150054, 0.0072543, -0.0211703, 0.0202900
8: -0.0035880, 0.0026439, -0.0069069, 0.0032611, -0.0068491, 0.0095509
9: -0.0126061, -0.0005802, -0.0138378, 0.0003756, -0.0129817, 0.0132576

Time for backsubstitution: 2.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 214

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0035818, upper bound: 0.0032868
time: 3.00 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0034506, upper bound: 0.0032687
time: 2.89 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.0058509, 0.0073957, 0.0057077, 0.0075763, -0.0017254, 0.0014947
1: -0.0005852, 0.0024069, -0.0018355, 0.0027566, -0.0029832, 0.0042424
2: -0.0024858, 0.0216484, -0.0053066, 0.0251068, -0.0275926, 0.0240623
3: -0.0044461, -0.0022905, -0.0046459, -0.0020386, -0.0024075, 0.0020857
4: -0.0002355, 0.0102228, -0.0012050, 0.0114452, -0.0116807, 0.0101194
5: -0.0020458, -0.0004846, -0.0022283, 0.0004570, -0.0025028, 0.0015566
6: 0.9911419, 0.9940053, 0.9887916, 0.9943398, -0.0028548, 0.0052136
7: -0.0138091, 0.0051222, -0.0159665, 0.0073349, -0.0188749, 0.0210888
8: -0.0033379, 0.0025931, -0.0098349, 0.0032863, -0.0059134, 0.0124280
9: -0.0125046, -0.0006671, -0.0138882, 0.0012188, -0.0137235, 0.0118023

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 214

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0036336, upper bound: 0.0032600
time: 2.93 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0034590, upper bound: 0.0032362
time: 2.27 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 5.67 + 599.09 = 604.76 seconds
