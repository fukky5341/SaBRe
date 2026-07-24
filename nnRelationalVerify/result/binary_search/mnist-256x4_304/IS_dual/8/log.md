## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 2000 seconds
Threshold: 43.1275729563
Search space: {k/256 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-24.6925583, 19.7640648, -24.6925583, 19.7640648, -44.4566231, 44.4566231)
1: (-22.1655941, 17.7110901, -22.1655941, 17.7110901, -39.8766861, 39.8766861)
2: (-28.0165939, 17.5855999, -28.0165939, 17.5855999, -45.6021957, 45.6021957)
3: (-30.1115532, 15.0882940, -30.1115532, 15.0882940, -45.1998444, 45.1998482)
4: (-28.4748173, 20.2006111, -28.4748173, 20.2006111, -48.6754303, 48.6754303)
5: (-24.4868717, 19.1075554, -24.4868717, 19.1075554, -43.5944290, 43.5944290)
6: (-22.5470924, 22.3522205, -22.5470924, 22.3522205, -44.8993111, 44.8993149)
7: (-24.8867416, 23.5568867, -24.8867416, 23.5568867, -48.4436264, 48.4436264)
8: (-34.8141861, 16.7162991, -34.8141861, 16.7162991, -51.5304832, 51.5304871)
9: (-21.9554176, 22.3245659, -21.9554176, 22.3245659, -44.2799835, 44.2799835)

## BASE Result
execution time: IAR + LP analysis = 1.41 + 8.04 = 9.45 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -43.1710632, upper bound: 43.1710632


# Binary Search by BASE starts (time budget: 1990.55 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=51.530487060546875
rel_dist={8: [-43.17090795605019, 43.17090794479026]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=51.530487060546875
rel_dist={8: [-43.17074367834749, 43.17074366907076]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=51.530487060546875
rel_dist={8: [-43.1705810903977, 43.17058109163787]}

## Binary Search Result
Binary search time: 36.42 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual) starts
Time budget: 1954.13 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1510648, upper bound: 43.1457939
time: 8.39 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1405668, upper bound: 43.1405668
time: 4.06 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 12.60 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 12.60
Output dim: 8, lower bound: -43.1510648, upper bound: 43.1457939
IS_A2, status: Status.UNKNOWN, split count: 1, time: 12.60
Output dim: 8, lower bound: -43.1405668, upper bound: 43.1405668

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -24.5925770, 19.6834145, -24.6925583, 19.7640648, -44.3566360, 44.3759727
1: -22.0777988, 17.6409893, -22.1655941, 17.7110901, -39.7888870, 39.8065834
2: -27.9050713, 17.5137882, -28.0165939, 17.5855999, -45.4906693, 45.5303802
3: -29.9917564, 15.0240593, -30.1115532, 15.0882940, -45.0800514, 45.1356087
4: -28.3656731, 20.1173515, -28.4748173, 20.2006111, -48.5662842, 48.5921707
5: -24.3891697, 19.0321083, -24.4868717, 19.1075554, -43.4967270, 43.5189819
6: -22.4538345, 22.2620869, -22.5470924, 22.3522205, -44.8060493, 44.8091812
7: -24.7865601, 23.4695358, -24.8867416, 23.5568867, -48.3434448, 48.3562775
8: -34.6859589, 16.6355858, -34.8141861, 16.7162991, -51.4022560, 51.4497719
9: -21.8643188, 22.2323341, -21.9554176, 22.3245659, -44.1888847, 44.1877518

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1340919, upper bound: 43.1329566
time: 9.30 seconds

## Relational analysis of IS_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 215

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1358870, upper bound: 43.1349223
time: 8.42 seconds

## Relational analysis of IS_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1489486, upper bound: 43.1441238
time: 8.56 seconds

## Relational analysis of IS_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1462336, upper bound: 43.1427493
time: 7.39 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1510648, upper bound: 43.1457939
time: 9.19 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -30.1759319, 24.0463200, -24.5919228, 19.6829967, -49.8589249, 48.6382370
1: -27.2746010, 21.5755043, -22.0773087, 17.6406345, -44.9152336, 43.6528130
2: -34.4703217, 21.2999153, -27.9044952, 17.5134850, -51.9838066, 49.2044106
3: -37.0141220, 17.9856491, -29.9910831, 15.0237474, -52.0378685, 47.9767303
4: -35.1700249, 24.5384579, -28.3650761, 20.1169548, -55.2869797, 52.9035339
5: -30.0841923, 23.3637733, -24.3887291, 19.0316792, -49.1158676, 47.7525024
6: -27.4695873, 27.2799320, -22.4533253, 22.2617111, -49.7313004, 49.7332573
7: -30.6063557, 29.0886269, -24.7861214, 23.4692211, -54.0755768, 53.8747482
8: -42.8730927, 19.6036835, -34.6854019, 16.6351147, -59.5082054, 54.2890816
9: -26.7513771, 27.1537323, -21.8638954, 22.2319336, -48.9833069, 49.0176277

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1364290, upper bound: 43.1368680
time: 6.20 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1361282, upper bound: 43.1361282
time: 4.76 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 30.47 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 30.47
Output dim: 8, lower bound: -43.1462336, upper bound: 43.1427493
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 30.47
Output dim: 8, lower bound: -43.1510648, upper bound: 43.1457939
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 30.47
Output dim: 8, lower bound: -43.1364290, upper bound: 43.1368680
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 30.47
Output dim: 8, lower bound: -43.1361282, upper bound: 43.1361282

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: -24.2663593, 19.4177074, -24.2903042, 19.4450378, -43.7113914, 43.7080116
1: -21.7665615, 17.4083061, -21.8066139, 17.4282684, -39.1948280, 39.2149200
2: -27.5068207, 17.2654533, -27.5614986, 17.3017693, -44.8085785, 44.8269501
3: -29.5370598, 14.8333769, -29.6202545, 14.8481913, -44.3852501, 44.4536285
4: -27.9432087, 19.8479042, -28.0166950, 19.8713856, -47.8145943, 47.8645935
5: -24.0439281, 18.7597733, -24.0903625, 18.7999134, -42.8438416, 42.8501358
6: -22.1359272, 21.9605427, -22.1759415, 21.9912338, -44.1271591, 44.1364784
7: -24.4344540, 23.1069870, -24.4799442, 23.1846714, -47.6191254, 47.5869255
8: -34.1633606, 16.4626408, -34.2710991, 16.4321918, -50.5955505, 50.7337379
9: -21.5716228, 21.9290237, -21.5955086, 21.9609833, -43.5325966, 43.5245285

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 214

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1404756, upper bound: 43.1373031
time: 9.57 seconds

## Relational analysis of IS_A1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of IS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 215

## Relational analysis of IS_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of IS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1417613, upper bound: 43.1386754
time: 6.17 seconds

## Relational analysis of IS_A1_A1_B2

### Relational analysis result of IS_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1403426, upper bound: 43.1370837
time: 9.94 seconds

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: -24.1127052, 19.3020153, -24.6925583, 19.7640648, -43.8767700, 43.9945755
1: -21.6485844, 17.3034744, -22.1655941, 17.7110901, -39.3596725, 39.4690704
2: -27.3606396, 17.1759300, -28.0165939, 17.5855999, -44.9462395, 45.1925201
3: -29.4061108, 14.7395582, -30.1115532, 15.0882940, -44.4944038, 44.8511086
4: -27.8169594, 19.7252350, -28.4748173, 20.2006111, -48.0175667, 48.2000504
5: -23.9155521, 18.6646671, -24.4868717, 19.1075554, -43.0231094, 43.1515388
6: -22.0118370, 21.8303471, -22.5470924, 22.3522205, -44.3640518, 44.3774414
7: -24.3002872, 23.0245667, -24.8867416, 23.5568867, -47.8571739, 47.9113045
8: -34.0344543, 16.2988338, -34.8141861, 16.7162991, -50.7507439, 51.1130180
9: -21.4342442, 21.7982864, -21.9554176, 22.3245659, -43.7588081, 43.7536964

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1340919, upper bound: 43.1329566
time: 8.25 seconds

## Relational analysis of IS_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1358870, upper bound: 43.1349223
time: 12.15 seconds

## Relational analysis of IS_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1489486, upper bound: 43.1441238
time: 7.93 seconds

## Relational analysis of IS_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 215

## Relational analysis of IS_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of IS_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1472883, upper bound: 43.1425586
time: 7.01 seconds

## Relational analysis of IS_A1_A2_B2

### Relational analysis result of IS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1466537, upper bound: 43.1413785
time: 7.74 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -30.1759319, 24.0463200, -22.4891777, 17.9982300, -48.1741562, 46.5354996
1: -27.2746010, 21.5755043, -20.2386780, 16.1712151, -43.4458122, 41.8141785
2: -34.4703217, 21.2999153, -25.5401859, 16.0127430, -50.4830627, 46.8401031
3: -37.0141220, 17.9856491, -27.4804001, 13.7239685, -50.7380905, 45.4660492
4: -35.1700249, 24.5384579, -26.0280190, 18.3728867, -53.5429115, 50.5664749
5: -30.0841923, 23.3637733, -22.3535767, 17.4388428, -47.5230331, 45.7173500
6: -27.4695873, 27.2799320, -20.4534702, 20.3756828, -47.8452644, 47.7334023
7: -30.6063557, 29.0886269, -22.6621017, 21.6154079, -52.2217636, 51.7507248
8: -42.8730927, 19.6036835, -31.9304352, 14.9694023, -57.8424950, 51.5341148
9: -26.7513771, 27.1537323, -19.9656639, 20.2973862, -47.0487556, 47.1193924

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of IS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 215

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1361282, upper bound: 43.1361282
time: 4.75 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1361282, upper bound: 43.1361282
time: 4.99 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -29.7216759, 23.6841736, -25.7311954, 20.5625191, -50.2841949, 49.4153671
1: -26.8771667, 21.2570076, -23.2082672, 18.4577103, -45.3348732, 44.4652748
2: -33.9604530, 20.9736977, -29.2517471, 18.2441444, -52.2045898, 50.2254448
3: -36.4709091, 17.7072563, -31.5171928, 15.6027641, -52.0736732, 49.2244492
4: -34.6659088, 24.1568546, -29.8188477, 20.9730453, -55.6389542, 53.9757004
5: -29.6471043, 23.0201397, -25.6105614, 19.9122658, -49.5593719, 48.6306992
6: -27.0396347, 26.8723946, -23.3950386, 23.2979965, -50.3376312, 50.2674332
7: -30.1456490, 28.6925449, -25.9573669, 24.7131634, -54.8588104, 54.6499100
8: -42.2808037, 19.2354889, -36.4193268, 17.0257950, -59.3065987, 55.6548080
9: -26.3412838, 26.7356396, -22.8381310, 23.1909008, -49.5321808, 49.5737686

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 215

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1361282, upper bound: 43.1361282
time: 5.42 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1361282, upper bound: 43.1361282
time: 4.21 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 51.42 seconds
IS_A1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 51.42
Output dim: 8, lower bound: -43.1417613, upper bound: 43.1386754
IS_A1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 51.42
Output dim: 8, lower bound: -43.1403426, upper bound: 43.1370837
IS_A1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 51.42
Output dim: 8, lower bound: -43.1472883, upper bound: 43.1425586
IS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 51.42
Output dim: 8, lower bound: -43.1466537, upper bound: 43.1413785
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 51.42
Output dim: 8, lower bound: -43.1361282, upper bound: 43.1361282
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 51.42
Output dim: 8, lower bound: -43.1361282, upper bound: 43.1361282
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 51.42
Output dim: 8, lower bound: -43.1361282, upper bound: 43.1361282
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 51.42
Output dim: 8, lower bound: -43.1361282, upper bound: 43.1361282

## BFS IS instance: IS_A1_A1_B1

### Backsubstitution after applying IS history:
0: -24.2663593, 19.4177074, -22.1964970, 17.7669449, -42.0333023, 41.6142044
1: -21.7665615, 17.4083061, -19.9741821, 15.9657125, -37.7322731, 37.3824883
2: -27.5068207, 17.2654533, -25.2062569, 15.8075504, -43.3143654, 42.4716988
3: -29.5370598, 14.8333769, -27.1179333, 13.5537577, -43.0908165, 41.9513092
4: -27.9432087, 19.8479042, -25.6887589, 18.1351223, -46.0783310, 45.5366592
5: -24.0439281, 18.7597733, -22.0627880, 17.2144203, -41.2583466, 40.8225632
6: -22.1359272, 21.9605427, -20.1840572, 20.1130409, -42.2489700, 42.1445999
7: -24.4344540, 23.1069870, -22.3638439, 21.3380623, -45.7725143, 45.4708290
8: -34.1633606, 16.4626408, -31.5259285, 14.7749138, -48.9382744, 47.9885635
9: -21.5716228, 21.9290237, -19.7049789, 20.0344353, -41.6060486, 41.6340027

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_A1_B1_A1

### Relational analysis result of IS_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1403426, upper bound: 43.1370837
time: 8.33 seconds

## Relational analysis of IS_A1_A1_B1_A2

### Relational analysis result of IS_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1403426, upper bound: 43.1370837
time: 7.85 seconds

## BFS IS instance: IS_A1_A1_B2

### Backsubstitution after applying IS history:
0: -23.7844162, 19.0323601, -25.4345188, 20.3272324, -44.1116371, 44.4668808
1: -21.3465881, 17.0703812, -22.9370136, 18.2483215, -39.5949020, 40.0073929
2: -26.9658451, 16.9207172, -28.9138680, 18.0350170, -45.0008545, 45.8345871
3: -28.9631882, 14.5356331, -31.1476135, 15.4289532, -44.3921432, 45.6832466
4: -27.4091415, 19.4438667, -29.4737930, 20.7320671, -48.1412086, 48.9176559
5: -23.5800362, 18.3961754, -25.3150635, 19.6845665, -43.2646027, 43.7112389
6: -21.6786556, 21.5284557, -23.1208801, 23.0302963, -44.7089462, 44.6493378
7: -23.9474277, 22.6862106, -25.6571274, 24.4308319, -48.3782578, 48.3433380
8: -33.5379105, 16.0718040, -36.0120125, 16.8263245, -50.3642311, 52.0838165
9: -21.1356182, 21.4844055, -22.5726776, 22.9236565, -44.0592728, 44.0570831

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 215

## Relational analysis of IS_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of IS_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of IS_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_A1_B2_A1

### Relational analysis result of IS_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1403426, upper bound: 43.1370837
time: 8.49 seconds

## Relational analysis of IS_A1_A1_B2_A2

### Relational analysis result of IS_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1403426, upper bound: 43.1370837
time: 11.20 seconds

## BFS IS instance: IS_A1_A2_B1

### Backsubstitution after applying IS history:
0: -24.1127052, 19.3020153, -22.5796375, 18.0712929, -42.1839981, 41.8816528
1: -21.6485844, 17.3034744, -20.3188419, 16.2338848, -37.8824615, 37.6223106
2: -27.3606396, 17.1759300, -25.6411686, 16.0775795, -43.4382172, 42.8170929
3: -29.4061108, 14.7395582, -27.5893879, 13.7822132, -43.1883240, 42.3289452
4: -27.8169594, 19.7252350, -26.1275845, 18.4474945, -46.2644539, 45.8528214
5: -23.9155521, 18.6646671, -22.4426041, 17.5068932, -41.4224472, 41.1072693
6: -22.0118370, 21.8303471, -20.5376549, 20.4574108, -42.4692421, 42.3679924
7: -24.3002872, 23.0245667, -22.7526894, 21.6950092, -45.9952965, 45.7772560
8: -34.0344543, 16.2988338, -32.0468483, 15.0409470, -49.0753975, 48.3456764
9: -21.4342442, 21.7982864, -20.0484657, 20.3805771, -41.8148193, 41.8467407

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 215

## Relational analysis of IS_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_A2_B1_A1

### Relational analysis result of IS_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1466163, upper bound: 43.1413785
time: 10.58 seconds

## Relational analysis of IS_A1_A2_B1_A2

### Relational analysis result of IS_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1466163, upper bound: 43.1413785
time: 7.17 seconds

## BFS IS instance: IS_A1_A2_B2

### Backsubstitution after applying IS history:
0: -23.6322632, 18.9182072, -25.8231430, 20.6362572, -44.2685204, 44.7413483
1: -21.2302551, 16.9672813, -23.2891731, 18.5215664, -39.7518234, 40.2564545
2: -26.8210239, 16.8322926, -29.3542862, 18.3099937, -45.1310158, 46.1865654
3: -28.8345108, 14.4438915, -31.6276073, 15.6618471, -44.4963531, 46.0714989
4: -27.2848225, 19.3229485, -29.9193459, 21.0485611, -48.3333817, 49.2422943
5: -23.4537277, 18.3020554, -25.7004471, 19.9812622, -43.4349899, 44.0025024
6: -21.5563717, 21.4002056, -23.4805374, 23.3807125, -44.9370842, 44.8807373
7: -23.8146458, 22.6056938, -26.0494213, 24.7937126, -48.6083450, 48.6551132
8: -33.4105797, 15.9100552, -36.5368118, 17.0983238, -50.5088997, 52.4468651
9: -21.0005245, 21.3555737, -22.9219646, 23.2748356, -44.2753601, 44.2775383

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_B2_B1

### Relational analysis result of IS_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1296981, upper bound: 43.1292533
time: 10.00 seconds

## Relational analysis of IS_A1_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 215

## Relational analysis of IS_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of IS_A1_A2_B2_B1

### Relational analysis result of IS_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1311807, upper bound: 43.1294366
time: 10.38 seconds

## Relational analysis of IS_A1_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_A2_B2_B1

### Relational analysis result of IS_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1430310, upper bound: 43.1383588
time: 8.67 seconds

## Relational analysis of IS_A1_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of IS_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_A2_B2_A1

### Relational analysis result of IS_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1466163, upper bound: 43.1413785
time: 8.52 seconds

## Relational analysis of IS_A1_A2_B2_A2

### Relational analysis result of IS_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1466163, upper bound: 43.1413785
time: 7.43 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -28.2203178, 22.4901791, -22.4891777, 17.9982300, -46.2185364, 44.9793549
1: -25.5578384, 20.2087517, -20.2386780, 16.1712151, -41.7290535, 40.4474297
2: -32.2777977, 19.9018517, -25.5401859, 16.0127430, -48.2905388, 45.4420395
3: -34.6678658, 16.7852802, -27.4804001, 13.7239685, -48.3918343, 44.2656784
4: -32.9896088, 22.9196358, -26.0280190, 18.3728867, -51.3624954, 48.9476547
5: -28.1890907, 21.8843651, -22.3535767, 17.4388428, -45.6279297, 44.2379417
6: -25.6103172, 25.5290947, -20.4534702, 20.3756828, -45.9859962, 45.9825630
7: -28.6222458, 27.3599873, -22.6621017, 21.6154079, -50.2376480, 50.0220871
8: -40.3059921, 18.0664539, -31.9304352, 14.9694023, -55.2753906, 49.9968872
9: -24.9904671, 25.3636894, -19.9656639, 20.2973862, -45.2878532, 45.3293495

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=51.530487060546875
rel_dist={8: [-43.17090795605019, 43.17090794479026]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1462040, upper bound: 43.1432586
time: 9.77 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1405371, upper bound: 43.1405371
time: 5.91 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 15.84 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 15.84
Output dim: 8, lower bound: -43.1462040, upper bound: 43.1432586
IS_A2, status: Status.UNKNOWN, split count: 1, time: 15.84
Output dim: 8, lower bound: -43.1405371, upper bound: 43.1405371

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -24.5925770, 19.6834145, -24.6925583, 19.7640648, -44.3566360, 44.3759727
1: -22.0777988, 17.6409893, -22.1655941, 17.7110901, -39.7888870, 39.8065834
2: -27.9050713, 17.5137882, -28.0165939, 17.5855999, -45.4906693, 45.5303802
3: -29.9917564, 15.0240593, -30.1115532, 15.0882940, -45.0800514, 45.1356087
4: -28.3656731, 20.1173515, -28.4748173, 20.2006111, -48.5662842, 48.5921707
5: -24.3891697, 19.0321083, -24.4868717, 19.1075554, -43.4967270, 43.5189819
6: -22.4538345, 22.2620869, -22.5470924, 22.3522205, -44.8060493, 44.8091812
7: -24.7865601, 23.4695358, -24.8867416, 23.5568867, -48.3434448, 48.3562775
8: -34.6859589, 16.6355858, -34.8141861, 16.7162991, -51.4022560, 51.4497719
9: -21.8643188, 22.2323341, -21.9554176, 22.3245659, -44.1888847, 44.1877518

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1424642, upper bound: 43.1398413
time: 8.09 seconds

## Relational analysis of IS_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1191710, upper bound: 43.1198871
time: 8.26 seconds

## Relational analysis of IS_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1312581, upper bound: 43.1305296
time: 9.41 seconds

## Relational analysis of IS_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1423561, upper bound: 43.1397341
time: 8.22 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1418054, upper bound: 43.1388138
time: 8.08 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -30.1759319, 24.0463200, -24.4273453, 19.5503883, -49.7263184, 48.4736633
1: -27.2746010, 21.5755043, -21.9331169, 17.5255318, -44.8001213, 43.5086212
2: -34.4703217, 21.2999153, -27.7210808, 17.3954201, -51.8657341, 49.0209885
3: -37.0141220, 17.9856491, -29.7942390, 14.9182901, -51.9324112, 47.7798882
4: -35.1700249, 24.5384579, -28.1854324, 19.9804230, -55.1504478, 52.7238922
5: -30.0841923, 23.3637733, -24.2285290, 18.9075184, -48.9916992, 47.5923004
6: -27.4695873, 27.2799320, -22.3000660, 22.1138401, -49.5834274, 49.5799980
7: -30.6063557, 29.0886269, -24.6215324, 23.3261070, -53.9324608, 53.7101593
8: -42.8730927, 19.6036835, -34.4746742, 16.5024700, -59.3755608, 54.0783539
9: -26.7513771, 27.1537323, -21.7141953, 22.0804710, -48.8318405, 48.8679276

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1362721, upper bound: 43.1365461
time: 5.74 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1361130, upper bound: 43.1361130
time: 4.52 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 40.88 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 40.88
Output dim: 8, lower bound: -43.1423561, upper bound: 43.1397341
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 40.88
Output dim: 8, lower bound: -43.1418054, upper bound: 43.1388138
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 40.88
Output dim: 8, lower bound: -43.1362721, upper bound: 43.1365461
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 40.88
Output dim: 8, lower bound: -43.1361130, upper bound: 43.1361130

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -23.8581505, 19.0968857, -22.5796375, 18.0712929, -41.9294434, 41.6765137
1: -21.4381142, 17.1260490, -20.3188419, 16.2338848, -37.6719894, 37.4448929
2: -27.0820808, 16.9896622, -25.6411686, 16.0775795, -43.1596603, 42.6308212
3: -29.1172447, 14.5733175, -27.5893879, 13.7822132, -42.8994522, 42.1627045
4: -27.5511265, 19.5019054, -26.1275845, 18.4474945, -45.9986191, 45.6294899
5: -23.6829643, 18.4758511, -22.4426041, 17.5068932, -41.1898575, 40.9184532
6: -21.7571964, 21.6044140, -20.5376549, 20.4574108, -42.2146072, 42.1420631
7: -24.0457268, 22.8281479, -22.7526894, 21.6950092, -45.7407379, 45.5808372
8: -33.7312317, 16.0447102, -32.0468483, 15.0409470, -48.7721748, 48.0915604
9: -21.2021923, 21.5579796, -20.0484657, 20.3805771, -41.5827675, 41.6064453

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1418012, upper bound: 43.1388138
time: 8.62 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1418012, upper bound: 43.1388138
time: 7.35 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -23.3956242, 18.7243195, -25.8231430, 20.6362572, -44.0318832, 44.5474625
1: -21.0354652, 16.8036671, -23.2891731, 18.5215664, -39.5570297, 40.0928421
2: -26.5597916, 16.6550674, -29.3542862, 18.3099937, -44.8697853, 46.0093498
3: -28.5677967, 14.2841454, -31.6276073, 15.6618471, -44.2296371, 45.9117508
4: -27.0393448, 19.1200085, -29.9193459, 21.0485611, -48.0878983, 49.0393524
5: -23.2361145, 18.1266727, -25.7004471, 19.9812622, -43.2173767, 43.8271179
6: -21.3157902, 21.1899719, -23.4805374, 23.3807125, -44.6965027, 44.6705093
7: -23.5769634, 22.4228611, -26.0494213, 24.7937126, -48.3706741, 48.4722824
8: -33.1261444, 15.6709023, -36.5368118, 17.0983238, -50.2244644, 52.2077141
9: -20.7829742, 21.1271591, -22.9219646, 23.2748356, -44.0578079, 44.0491257

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1418012, upper bound: 43.1388138
time: 9.52 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1418012, upper bound: 43.1388138
time: 9.77 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -29.4985905, 23.5063915, -22.3406715, 17.8783684, -47.3769608, 45.8470573
1: -26.6804352, 21.1002731, -20.1069565, 16.0682411, -42.7486687, 41.2072296
2: -33.7102852, 20.8149662, -25.3742085, 15.9062481, -49.6165314, 46.1891747
3: -36.2028198, 17.5706005, -27.3010139, 13.6284275, -49.8312454, 44.8716125
4: -34.4177399, 23.9708748, -25.8642807, 18.2503967, -52.6681366, 49.8351555
5: -29.4301739, 22.8511238, -22.2071514, 17.3274460, -46.7576180, 45.0582657
6: -26.8283062, 26.6721478, -20.3151913, 20.2415810, -47.0698814, 46.9873352
7: -29.9207573, 28.4949512, -22.5130692, 21.4846039, -51.4053612, 51.0080185
8: -41.9859772, 19.0599823, -31.7390766, 14.8520393, -56.8380165, 50.7990570
9: -26.1418724, 26.5310059, -19.8298435, 20.1608467, -46.3027115, 46.3608475

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1361130, upper bound: 43.1361130
time: 5.35 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1361130, upper bound: 43.1361130
time: 12.28 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -29.0621719, 23.1575890, -25.5755291, 20.4365921, -49.4987488, 48.7331161
1: -26.3005505, 20.7964172, -23.0684471, 18.3482170, -44.6487656, 43.8648605
2: -33.2202530, 20.4993362, -29.0784111, 18.1314926, -51.3517456, 49.5777473
3: -35.6834183, 17.3011360, -31.3271523, 15.5016384, -51.1850586, 48.6282883
4: -33.9348640, 23.6097336, -29.6467896, 20.8445110, -54.7793694, 53.2565193
5: -29.0095062, 22.5212231, -25.4572430, 19.7949333, -48.8044395, 47.9784660
6: -26.4127541, 26.2822952, -23.2493610, 23.1563129, -49.5690613, 49.5316544
7: -29.4751892, 28.1121597, -25.8028793, 24.5750523, -54.0502396, 53.9150391
8: -41.4165039, 18.7087269, -36.2212524, 16.9015350, -58.3180351, 54.9299774
9: -25.7471390, 26.1282425, -22.6946526, 23.0477772, -48.7949142, 48.8228951

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1361130, upper bound: 43.1361130
time: 5.54 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1361130, upper bound: 43.1361130
time: 7.74 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 19.02 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 19.02
Output dim: 8, lower bound: -43.1418012, upper bound: 43.1388138
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 19.02
Output dim: 8, lower bound: -43.1418012, upper bound: 43.1388138
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 19.02
Output dim: 8, lower bound: -43.1418012, upper bound: 43.1388138
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 19.02
Output dim: 8, lower bound: -43.1418012, upper bound: 43.1388138
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 19.02
Output dim: 8, lower bound: -43.1361130, upper bound: 43.1361130
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 19.02
Output dim: 8, lower bound: -43.1361130, upper bound: 43.1361130
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 19.02
Output dim: 8, lower bound: -43.1361130, upper bound: 43.1361130
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 19.02
Output dim: 8, lower bound: -43.1361130, upper bound: 43.1361130

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -22.4901924, 17.9989300, -22.5796375, 18.0712929, -40.5614853, 40.5785599
1: -20.2394505, 16.1718464, -20.3188419, 16.2338848, -36.4733353, 36.4906845
2: -25.5411358, 16.0133228, -25.6411686, 16.0775795, -41.6187096, 41.6544800
3: -27.4814758, 13.7244577, -27.5893879, 13.7822132, -41.2636871, 41.3138466
4: -26.0290070, 18.3735981, -26.1275845, 18.4474945, -44.4765015, 44.5011787
5: -22.3544273, 17.4395237, -22.4426041, 17.5068932, -39.8613205, 39.8821259
6: -20.4543400, 20.3763466, -20.5376549, 20.4574108, -40.9117470, 40.9139938
7: -22.6629200, 21.6160870, -22.7526894, 21.6950092, -44.3579292, 44.3687744
8: -31.9315414, 14.9701138, -32.0468483, 15.0409470, -46.9724884, 47.0169601
9: -19.9664116, 20.2980652, -20.0484657, 20.3805771, -40.3469887, 40.3465271

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1356467, upper bound: 43.1337941
time: 8.97 seconds

## Relational analysis of IS_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 215

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1279181, upper bound: 43.1243266
time: 6.92 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1333461, upper bound: 43.1299807
time: 8.02 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -25.7264938, 20.5578270, -22.5796375, 18.0712929, -43.7977829, 43.1374588
1: -23.2013054, 18.4529724, -20.3188419, 16.2338848, -39.4351883, 38.7718086
2: -29.2467308, 18.2393532, -25.6411686, 16.0775795, -45.3243065, 43.8805161
3: -31.5087032, 15.5984926, -27.5893879, 13.7822132, -45.2909164, 43.1878777
4: -29.8120251, 20.9687347, -26.1275845, 18.4474945, -48.2595177, 47.0963173
5: -25.6051254, 19.9080658, -22.4426041, 17.5068932, -43.1120071, 42.3506699
6: -23.3896275, 23.2919960, -20.5376549, 20.4574108, -43.8470383, 43.8296509
7: -25.9538460, 24.7073364, -22.7526894, 21.6950092, -47.6488457, 47.4600258
8: -36.4148865, 17.0203018, -32.0468483, 15.0409470, -51.4558258, 49.0671463
9: -22.8319721, 23.1858959, -20.0484657, 20.3805771, -43.2125473, 43.2343597

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1356467, upper bound: 43.1337941
time: 8.75 seconds

## Relational analysis of IS_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A1_B1_A2_A1

### Relational analysis result of IS_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1289488, upper bound: 43.1265785
time: 7.91 seconds

## Relational analysis of IS_A1_B1_A2_A2

### Relational analysis result of IS_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1333461, upper bound: 43.1299808
time: 13.95 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -22.4901924, 17.9989300, -25.8231430, 20.6362572, -43.1264496, 43.8220749
1: -20.2394505, 16.1718464, -23.2891731, 18.5215664, -38.7610168, 39.4610176
2: -25.5411358, 16.0133228, -29.3542862, 18.3099937, -43.8511238, 45.3675957
3: -27.4814758, 13.7244577, -31.6276073, 15.6618471, -43.1433182, 45.3520622
4: -26.0290070, 18.3735981, -29.9193459, 21.0485611, -47.0775604, 48.2929459
5: -22.3544273, 17.4395237, -25.7004471, 19.9812622, -42.3356895, 43.1399689
6: -20.4543400, 20.3763466, -23.4805374, 23.3807125, -43.8350525, 43.8568726
7: -22.6629200, 21.6160870, -26.0494213, 24.7937126, -47.4566345, 47.6655045
8: -31.9315414, 14.9701138, -36.5368118, 17.0983238, -49.0298653, 51.5069237
9: -19.9664116, 20.2980652, -22.9219646, 23.2748356, -43.2412491, 43.2200317

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1336720, upper bound: 43.1314365
time: 9.96 seconds

## Relational analysis of IS_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1268875, upper bound: 43.1229023
time: 7.56 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1326738, upper bound: 43.1288669
time: 6.99 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -25.7261734, 20.5575562, -25.8231430, 20.6362572, -46.3624306, 46.3806992
1: -23.2010098, 18.4527607, -23.2891731, 18.5215664, -41.7225723, 41.7419281
2: -29.2463169, 18.2391052, -29.3542862, 18.3099937, -47.5563126, 47.5933800
3: -31.5083466, 15.5983286, -31.6276073, 15.6618471, -47.1701889, 47.2259331
4: -29.8116913, 20.9685211, -29.9193459, 21.0485611, -50.8602524, 50.8878670
5: -25.6048012, 19.9078598, -25.7004471, 19.9812622, -45.5860596, 45.6083031
6: -23.3892860, 23.2917366, -23.4805374, 23.3807125, -46.7699966, 46.7722702
7: -25.9534721, 24.7069702, -26.0494213, 24.7937126, -50.7471771, 50.7563934
8: -36.4145126, 17.0200825, -36.5368118, 17.0983238, -53.5128326, 53.5568886
9: -22.8316994, 23.1856918, -22.9219646, 23.2748356, -46.1065369, 46.1076584

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1336720, upper bound: 43.1314365
time: 8.42 seconds

## Relational analysis of IS_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 215

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1268875, upper bound: 43.1229023
time: 12.44 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1326738, upper bound: 43.1288669
time: 9.80 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -28.2203178, 22.4901791, -22.3406715, 17.8783684, -46.0986824, 44.8308487
1: -25.5578384, 20.2087517, -20.1069565, 16.0682411, -41.6260719, 40.3157082
2: -32.2777977, 19.9018517, -25.3742085, 15.9062481, -48.1840439, 45.2760620
3: -34.6678658, 16.7852802, -27.3010139, 13.6284275, -48.2962914, 44.0862961
4: -32.9896088, 22.9196358, -25.8642807, 18.2503967, -51.2400055, 48.7839127
5: -28.1890907, 21.8843651, -22.2071514, 17.3274460, -45.5165367, 44.0915146
6: -25.6103172, 25.5290947, -20.3151913, 20.2415810, -45.8518944, 45.8442841
7: -28.6222458, 27.3599873, -22.5130692, 21.4846039, -50.1068497, 49.8730545
8: -40.3059921, 18.0664539, -31.7390766, 14.8520393, -55.1580276, 49.8055305
9: -24.9904671, 25.3636894, -19.8298435, 20.1608467, -45.1513138, 45.1935349

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=51.530487060546875
rel_dist={8: [-43.17074367834749, 43.17074366907076]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1426517, upper bound: 43.1414379
time: 11.07 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1404961, upper bound: 43.1404961
time: 6.58 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 17.80 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 17.80
Output dim: 8, lower bound: -43.1426517, upper bound: 43.1414379
IS_A2, status: Status.UNKNOWN, split count: 1, time: 17.80
Output dim: 8, lower bound: -43.1404961, upper bound: 43.1404961

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -24.5925770, 19.6834145, -24.6925583, 19.7640648, -44.3566360, 44.3759727
1: -22.0777988, 17.6409893, -22.1655941, 17.7110901, -39.7888870, 39.8065834
2: -27.9050713, 17.5137882, -28.0165939, 17.5855999, -45.4906693, 45.5303802
3: -29.9917564, 15.0240593, -30.1115532, 15.0882940, -45.0800514, 45.1356087
4: -28.3656731, 20.1173515, -28.4748173, 20.2006111, -48.5662842, 48.5921707
5: -24.3891697, 19.0321083, -24.4868717, 19.1075554, -43.4967270, 43.5189819
6: -22.4538345, 22.2620869, -22.5470924, 22.3522205, -44.8060493, 44.8091812
7: -24.7865601, 23.4695358, -24.8867416, 23.5568867, -48.3434448, 48.3562775
8: -34.6859589, 16.6355858, -34.8141861, 16.7162991, -51.4022560, 51.4497719
9: -21.8643188, 22.2323341, -21.9554176, 22.3245659, -44.1888847, 44.1877518

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 214

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 148

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1221034, upper bound: 43.1203164
time: 8.69 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1213044, upper bound: 43.1199087
time: 8.24 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -30.1759319, 24.0463200, -23.9070244, 19.1301270, -49.3060608, 47.9533463
1: -27.2746010, 21.5755043, -21.4758968, 17.1616077, -44.4362106, 43.0513992
2: -34.4703217, 21.2999153, -27.1398125, 17.0214996, -51.4918175, 48.4397240
3: -37.0141220, 17.9856491, -29.1704788, 14.5847731, -51.5988960, 47.1561279
4: -35.1700249, 24.5384579, -27.6155128, 19.5490551, -54.7190781, 52.1539688
5: -30.0841923, 23.3637733, -23.7208672, 18.5152836, -48.5994720, 47.0846329
6: -27.4695873, 27.2799320, -21.8152142, 21.6455078, -49.1150970, 49.0951462
7: -30.6063557, 29.0886269, -24.1010361, 22.8723488, -53.4786987, 53.1896629
8: -42.8730927, 19.6036835, -33.8067284, 16.0833130, -58.9564056, 53.4104080
9: -26.7513771, 27.1537323, -21.2407074, 21.6004276, -48.3517990, 48.3944397

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 148

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1191815, upper bound: 43.1198746
time: 6.17 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1185205, upper bound: 43.1185205
time: 6.62 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 22.68 seconds
IS_A1_A1, status: Status.VERIFIED, split count: 2, time: 22.68
Output dim: 8, lower bound: -43.1221034, upper bound: 43.1203164
IS_A1_A2, status: Status.VERIFIED, split count: 2, time: 22.68
Output dim: 8, lower bound: -43.1213044, upper bound: 43.1199087
IS_A2_B1, status: Status.VERIFIED, split count: 2, time: 22.68
Output dim: 8, lower bound: -43.1191815, upper bound: 43.1198746
IS_A2_B2, status: Status.VERIFIED, split count: 2, time: 22.68
Output dim: 8, lower bound: -43.1185205, upper bound: 43.1185205
Binary search (step 2): status=Status.VERIFIED, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=51.530487060546875
rel_dist={8: [-43.1705810903977, 43.17058109163787]}

## Binary search (step 3) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1445238, upper bound: 43.1423612
time: 8.49 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1405175, upper bound: 43.1405175
time: 8.41 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 17.06 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 17.06
Output dim: 8, lower bound: -43.1445238, upper bound: 43.1423612
IS_A2, status: Status.UNKNOWN, split count: 1, time: 17.06
Output dim: 8, lower bound: -43.1405175, upper bound: 43.1405175

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -24.5925770, 19.6834145, -24.6925583, 19.7640648, -44.3566360, 44.3759727
1: -22.0777988, 17.6409893, -22.1655941, 17.7110901, -39.7888870, 39.8065834
2: -27.9050713, 17.5137882, -28.0165939, 17.5855999, -45.4906693, 45.5303802
3: -29.9917564, 15.0240593, -30.1115532, 15.0882940, -45.0800514, 45.1356087
4: -28.3656731, 20.1173515, -28.4748173, 20.2006111, -48.5662842, 48.5921707
5: -24.3891697, 19.0321083, -24.4868717, 19.1075554, -43.4967270, 43.5189819
6: -22.4538345, 22.2620869, -22.5470924, 22.3522205, -44.8060493, 44.8091812
7: -24.7865601, 23.4695358, -24.8867416, 23.5568867, -48.3434448, 48.3562775
8: -34.6859589, 16.6355858, -34.8141861, 16.7162991, -51.4022560, 51.4497719
9: -21.8643188, 22.2323341, -21.9554176, 22.3245659, -44.1888847, 44.1877518

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1284077, upper bound: 43.1255918
time: 7.97 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1346978, upper bound: 43.1322492
time: 9.45 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -30.1759319, 24.0463200, -24.2782497, 19.4302139, -49.6061478, 48.3245697
1: -27.2746010, 21.5755043, -21.8024292, 17.4212723, -44.6958694, 43.3779297
2: -34.4703217, 21.2999153, -27.5547752, 17.2883415, -51.7586517, 48.8546829
3: -37.0141220, 17.9856491, -29.6158352, 14.8227272, -51.8368492, 47.6014862
4: -35.1700249, 24.5384579, -28.0225010, 19.8568401, -55.0268631, 52.5609550
5: -30.0841923, 23.3637733, -24.0834160, 18.7950039, -48.8791924, 47.4471893
6: -27.4695873, 27.2799320, -22.1611652, 21.9798546, -49.4494400, 49.4410973
7: -30.6063557, 29.0886269, -24.4723892, 23.1964073, -53.8027611, 53.5610161
8: -42.8730927, 19.6036835, -34.2837067, 16.3823719, -59.2554588, 53.8873863
9: -26.7513771, 27.1537323, -21.5785732, 21.9431725, -48.6945419, 48.7323074

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1362098, upper bound: 43.1363988
time: 6.21 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1361045, upper bound: 43.1361045
time: 6.29 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 50.79 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 50.79
Output dim: 8, lower bound: -43.1284077, upper bound: 43.1255918
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 50.79
Output dim: 8, lower bound: -43.1346978, upper bound: 43.1322492
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 50.79
Output dim: 8, lower bound: -43.1362098, upper bound: 43.1363988
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 50.79
Output dim: 8, lower bound: -43.1361045, upper bound: 43.1361045

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -20.1840019, 16.1865921, -16.8249416, 13.5226021, -33.7066002, 33.0115318
1: -18.1691837, 14.5486736, -15.1271172, 12.1715088, -30.3406925, 29.6757870
2: -22.9275055, 14.3799610, -19.1207237, 12.0093746, -34.9368744, 33.5006828
3: -24.6336555, 12.2593145, -20.5349598, 10.2281017, -34.8617554, 32.7942734
4: -23.4591160, 16.4849777, -19.6008205, 13.7641506, -37.2232628, 36.0858002
5: -20.0814781, 15.7079802, -16.7620621, 13.1661930, -33.2476730, 32.4700432
6: -18.3819237, 18.3271217, -15.3057022, 15.3110809, -33.6929970, 33.6328201
7: -20.3517342, 19.5144348, -16.9279099, 16.4073772, -36.7591095, 36.4423409
8: -28.9298935, 13.2861166, -24.4201012, 10.9323349, -39.8622208, 37.7062149
9: -17.9308395, 18.2677898, -14.9432278, 15.2506809, -33.1815186, 33.2110176

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1279253, upper bound: 43.1252734
time: 8.44 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1279253, upper bound: 43.1255918
time: 8.54 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -22.1451702, 17.7364483, -20.5827618, 16.4996567, -38.6448288, 38.3192101
1: -19.9177971, 15.9230356, -18.5231171, 14.8211727, -34.7389679, 34.4461517
2: -25.1526947, 15.7822905, -23.3871269, 14.6775570, -39.8302422, 39.1694183
3: -27.0235577, 13.5002689, -25.1150188, 12.5148745, -39.5384331, 38.6152802
4: -25.6516857, 18.0902748, -23.8989639, 16.8061142, -42.4578018, 41.9892387
5: -21.9984589, 17.1890697, -20.4635334, 16.0119934, -38.0104485, 37.6526031
6: -20.1884460, 20.0779896, -18.7411003, 18.6813583, -38.8698044, 38.8190804
7: -22.3334846, 21.2933006, -20.7505188, 19.8872986, -42.2207794, 42.0438194
8: -31.5117931, 14.7552996, -29.4736805, 13.5860252, -45.0978165, 44.2289696
9: -19.6711960, 20.0165596, -18.2791023, 18.6082859, -38.2794762, 38.2956619

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1293323, upper bound: 43.1277751
time: 8.17 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1293323, upper bound: 43.1322491
time: 10.45 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -29.1277428, 23.2095013, -22.2088642, 17.7720795, -46.8998222, 45.4183655
1: -26.3555679, 20.8405743, -19.9899731, 15.9764118, -42.3319778, 40.8305473
2: -33.2934074, 20.5489197, -25.2266121, 15.8117285, -49.1051331, 45.7755318
3: -35.7579994, 17.3418446, -27.1410255, 13.5437222, -49.3017197, 44.4828644
4: -34.0054245, 23.6641140, -25.7182007, 18.1415138, -52.1469383, 49.3823166
5: -29.0710640, 22.5696678, -22.0769501, 17.2288494, -46.2999115, 44.6466179
6: -26.4756317, 26.3399563, -20.1921272, 20.1229553, -46.5985870, 46.5320816
7: -29.5432892, 28.1668930, -22.3798542, 21.3681793, -50.9114685, 50.5467453
8: -41.4967232, 18.7671394, -31.5688305, 14.7479753, -56.2446938, 50.3359680
9: -25.8080444, 26.1902618, -19.7094154, 20.0395451, -45.8475800, 45.8996620

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 214

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1361045, upper bound: 43.1361045
time: 7.14 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1361045, upper bound: 43.1361045
time: 7.40 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -28.7525711, 22.9125366, -25.4392872, 20.3270130, -49.0795822, 48.3518181
1: -26.0289574, 20.5810165, -22.9483681, 18.2535858, -44.2825432, 43.5293846
2: -32.8739738, 20.2775459, -28.9263401, 18.0340347, -50.9080048, 49.2038879
3: -35.3133545, 17.1111755, -31.1631966, 15.4139767, -50.7273254, 48.2743721
4: -33.5892487, 23.3560448, -29.4975758, 20.7321529, -54.3214035, 52.8536224
5: -28.7095871, 22.2875690, -25.3237381, 19.6926575, -48.4022408, 47.6112976
6: -26.1170807, 26.0060387, -23.1227779, 23.0335674, -49.1506424, 49.1288147
7: -29.1604595, 27.8382797, -25.6664505, 24.4556675, -53.6161232, 53.5047226
8: -41.0124626, 18.4667778, -36.0463448, 16.7940369, -57.8064995, 54.5131226
9: -25.4675770, 25.8460083, -22.5704651, 22.9227943, -48.3903694, 48.4164696

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1361045, upper bound: 43.1361045
time: 5.55 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1361045, upper bound: 43.1361045
time: 6.18 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 13.18 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 13.18
Output dim: 8, lower bound: -43.1279253, upper bound: 43.1252734
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 13.18
Output dim: 8, lower bound: -43.1279253, upper bound: 43.1255918
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 13.18
Output dim: 8, lower bound: -43.1293323, upper bound: 43.1277751
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 13.18
Output dim: 8, lower bound: -43.1293323, upper bound: 43.1322491
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 13.18
Output dim: 8, lower bound: -43.1361045, upper bound: 43.1361045
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 13.18
Output dim: 8, lower bound: -43.1361045, upper bound: 43.1361045
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 13.18
Output dim: 8, lower bound: -43.1361045, upper bound: 43.1361045
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 13.18
Output dim: 8, lower bound: -43.1361045, upper bound: 43.1361045

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -16.7589626, 13.4682751, -16.8249416, 13.5226021, -30.2815590, 30.2932167
1: -15.0670795, 12.1240711, -15.1271172, 12.1715088, -27.2385883, 27.2511883
2: -19.0459061, 11.9617844, -19.1207237, 12.0093746, -31.0552788, 31.0825081
3: -20.4547195, 10.1863060, -20.5349598, 10.2281017, -30.6828213, 30.7212658
4: -19.5249634, 13.7101040, -19.6008205, 13.7641506, -33.2891083, 33.3109207
5: -16.6963921, 13.1158695, -16.7620621, 13.1661930, -29.8625851, 29.8779297
6: -15.2436523, 15.2509050, -15.3057022, 15.3110809, -30.5547295, 30.5566044
7: -16.8600655, 16.3474846, -16.9279099, 16.4073772, -33.2674408, 33.2753906
8: -24.3326035, 10.8815794, -24.4201012, 10.9323349, -35.2649384, 35.3016777
9: -14.8829231, 15.1881247, -14.9432278, 15.2506809, -30.1336040, 30.1313515

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1183666, upper bound: 43.1169261
time: 8.73 seconds

## Relational analysis of IS_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1247671, upper bound: 43.1225325
time: 8.59 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1237544, upper bound: 43.1212156
time: 7.09 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -20.4943218, 16.4272060, -16.8249416, 13.5226021, -34.0169144, 33.2521439
1: -18.4439697, 14.7575264, -15.1271172, 12.1715088, -30.6154785, 29.8846436
2: -23.2875462, 14.6134043, -19.1207237, 12.0093746, -35.2969208, 33.7341270
3: -25.0005016, 12.4487619, -20.5349598, 10.2281017, -35.2286034, 32.9837227
4: -23.8018017, 16.7306042, -19.6008205, 13.7641506, -37.5659485, 36.3314247
5: -20.3772278, 15.9434910, -16.7620621, 13.1661930, -33.5434189, 32.7055511
6: -18.6554775, 18.6008873, -15.3057022, 15.3110809, -33.9665604, 33.9065857
7: -20.6613464, 19.8090096, -16.9279099, 16.4073772, -37.0687256, 36.7369156
8: -29.3610573, 13.5126705, -24.4201012, 10.9323349, -40.2933922, 37.9327698
9: -18.1978054, 18.5253906, -14.9432278, 15.2506809, -33.4484749, 33.4686165

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A2_A1

### Relational analysis result of IS_A1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1238815, upper bound: 43.1216715
time: 8.71 seconds

## Relational analysis of IS_A1_B1_A2_A2

### Relational analysis result of IS_A1_B1_A2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1237544, upper bound: 43.1216234
time: 8.51 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -16.7589626, 13.4682751, -20.5827618, 16.4996567, -33.2586174, 34.0510368
1: -15.0670795, 12.1240711, -18.5231171, 14.8211727, -29.8882523, 30.6471844
2: -19.0459061, 11.9617844, -23.3871269, 14.6775570, -33.7234612, 35.3489113
3: -20.4547195, 10.1863060, -25.1150188, 12.5148745, -32.9695930, 35.3013229
4: -19.5249634, 13.7101040, -23.8989639, 16.8061142, -36.3310776, 37.6090622
5: -16.6963921, 13.1158695, -20.4635334, 16.0119934, -32.7083778, 33.5793991
6: -15.2436523, 15.2509050, -18.7411003, 18.6813583, -33.9250107, 33.9920044
7: -16.8600655, 16.3474846, -20.7505188, 19.8872986, -36.7473602, 37.0980034
8: -24.3326035, 10.8815794, -29.4736805, 13.5860252, -37.9186287, 40.3552513
9: -14.8829231, 15.1881247, -18.2791023, 18.6082859, -33.4912071, 33.4672279

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1183666, upper bound: 43.1201708
time: 8.77 seconds

## Relational analysis of IS_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1247671, upper bound: 43.1246791
time: 9.61 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1237544, upper bound: 43.1240302
time: 17.73 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -20.5062866, 16.4373322, -20.5827618, 16.4996567, -37.0059433, 37.0200958
1: -18.4539566, 14.7669249, -18.5231171, 14.8211727, -33.2751312, 33.2900391
2: -23.3006783, 14.6218462, -23.3871269, 14.6775570, -37.9782333, 38.0089722
3: -25.0216923, 12.4649935, -25.1150188, 12.5148745, -37.5365639, 37.5800133
4: -23.8129597, 16.7422562, -23.8989639, 16.8061142, -40.6190720, 40.6412201
5: -20.3877869, 15.9539852, -20.4635334, 16.0119934, -36.3997765, 36.4175186
6: -18.6691685, 18.6115875, -18.7411003, 18.6813583, -37.3505249, 37.3526802
7: -20.6719627, 19.8185387, -20.7505188, 19.8872986, -40.5592575, 40.5690575
8: -29.3734818, 13.5254469, -29.4736805, 13.5860252, -42.9595070, 42.9991264
9: -18.2090111, 18.5367432, -18.2791023, 18.6082859, -36.8172951, 36.8158417

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1183666, upper bound: 43.1271052
time: 8.56 seconds

## Relational analysis of IS_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1237542, upper bound: 43.1303776
time: 8.97 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1210078, upper bound: 43.1179252
time: 64.12 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -28.2203178, 22.4901791, -22.2088642, 17.7720795, -45.9923973, 44.6990433
1: -25.5578384, 20.2087517, -19.9899731, 15.9764118, -41.5342484, 40.1987228
2: -32.2777977, 19.9018517, -25.2266121, 15.8117285, -48.0895195, 45.1284561
3: -34.6678658, 16.7852802, -27.1410255, 13.5437222, -48.2115860, 43.9263000
4: -32.9896088, 22.9196358, -25.7182007, 18.1415138, -51.1311226, 48.6378365
5: -28.1890907, 21.8843651, -22.0769501, 17.2288494, -45.4179382, 43.9613152
6: -25.6103172, 25.5290947, -20.1921272, 20.1229553, -45.7332726, 45.7212219
7: -28.6222458, 27.3599873, -22.3798542, 21.3681793, -49.9904251, 49.7398415
8: -40.3059921, 18.0664539, -31.5688305, 14.7479753, -55.0539627, 49.6352844
9: -24.9904671, 25.3636894, -19.7094154, 20.0395451, -45.0300064, 45.0730972

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 214

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1199444, upper bound: 43.1197968
time: 5.71 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1258551, upper bound: 43.1260588
time: 6.24 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -31.5945549, 25.1628647, -22.2088642, 17.7720795, -49.3666344, 47.3717270
1: -28.6641083, 22.5896492, -19.9899731, 15.9764118, -44.6405182, 42.5796204
2: -36.1624641, 22.2232304, -25.2266121, 15.8117285, -51.9741859, 47.4498329
3: -38.8779907, 18.7360210, -27.1410255, 13.5437222, -52.4217148, 45.8770447
4: -36.9562645, 25.6326675, -25.7182007, 18.1415138, -55.0977783, 51.3508682
5: -31.5915909, 24.4638405, -22.0769501, 17.2288494, -48.8204384, 46.5407906
6: -28.6758327, 28.5737953, -20.1921272, 20.1229553, -48.7987900, 48.7659225
7: -32.0679245, 30.6090279, -22.3798542, 21.3681793, -53.4361038, 52.9888840
8: -45.0140991, 20.1968174, -31.5688305, 14.7479753, -59.7620735, 51.7656479
9: -27.9875793, 28.3774071, -19.7094154, 20.0395451, -48.0271225, 48.0868149

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 214

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 3): status=Status.UNKNOWN, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=51.530487060546875
rel_dist={8: [-43.17066424353937, 43.17066423708645]}

## Binary Search with IS_dual Result
status: Status.VERIFIED
Maximum delta epsilon: 0.00390625
execution time: 1881.92 seconds
