## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 8)
Time budget: 600 seconds
Split limit: 100
Threshold: 43.1275729563


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

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.00 + 12.39 = 13.39 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -43.1707437, upper bound: 43.1707437

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
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

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1462040, upper bound: 43.1432586
time: 9.43 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1405371, upper bound: 43.1405371
time: 5.70 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 15.23 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 15.23
Output dim: 8, lower bound: -43.1462040, upper bound: 43.1432586
NS_A2, status: Status.UNKNOWN, split count: 1, time: 15.23
Output dim: 8, lower bound: -43.1405371, upper bound: 43.1405371

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
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

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 102

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 161

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 153

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 196

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 77

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1424642, upper bound: 43.1398413
time: 7.90 seconds

## Relational analysis of NS_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1191710, upper bound: 43.1198871
time: 7.92 seconds

## Relational analysis of NS_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 219

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1312581, upper bound: 43.1305296
time: 8.99 seconds

## Relational analysis of NS_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 95

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1423561, upper bound: 43.1397341
time: 7.76 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1418054, upper bound: 43.1388138
time: 7.70 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
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

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of NS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of NS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 102

### Candidate
type: A, layer: 1, pos: 153

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of NS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 161

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of NS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of NS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of NS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 77

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of NS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 196

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of NS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1362721, upper bound: 43.1365461
time: 5.56 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1361130, upper bound: 43.1361130
time: 4.31 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 27.90 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 27.90
Output dim: 8, lower bound: -43.1423561, upper bound: 43.1397341
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 27.90
Output dim: 8, lower bound: -43.1418054, upper bound: 43.1388138
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 27.90
Output dim: 8, lower bound: -43.1362721, upper bound: 43.1365461
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 27.90
Output dim: 8, lower bound: -43.1361130, upper bound: 43.1361130

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
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

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1418012, upper bound: 43.1388138
time: 8.32 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1418012, upper bound: 43.1388138
time: 7.04 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
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

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1418012, upper bound: 43.1388138
time: 9.15 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1418012, upper bound: 43.1388138
time: 9.41 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
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

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1361130, upper bound: 43.1361130
time: 5.19 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1361130, upper bound: 43.1361130
time: 11.88 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
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

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 102

### Candidate
type: B, layer: 1, pos: 153

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1361130, upper bound: 43.1361130
time: 5.28 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1361130, upper bound: 43.1361130
time: 7.47 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 13.74 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 13.74
Output dim: 8, lower bound: -43.1418012, upper bound: 43.1388138
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 13.74
Output dim: 8, lower bound: -43.1418012, upper bound: 43.1388138
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 13.74
Output dim: 8, lower bound: -43.1418012, upper bound: 43.1388138
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 13.74
Output dim: 8, lower bound: -43.1418012, upper bound: 43.1388138
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 13.74
Output dim: 8, lower bound: -43.1361130, upper bound: 43.1361130
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 13.74
Output dim: 8, lower bound: -43.1361130, upper bound: 43.1361130
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 13.74
Output dim: 8, lower bound: -43.1361130, upper bound: 43.1361130
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 13.74
Output dim: 8, lower bound: -43.1361130, upper bound: 43.1361130

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
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

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 102

### Candidate
type: A, layer: 1, pos: 102

### Candidate
type: B, layer: 1, pos: 153

### Candidate
type: A, layer: 1, pos: 153

### Candidate
type: B, layer: 1, pos: 161

### Candidate
type: A, layer: 1, pos: 161

### Candidate
type: B, layer: 1, pos: 77

### Candidate
type: A, layer: 1, pos: 77

### Candidate
type: B, layer: 1, pos: 196

### Candidate
type: A, layer: 1, pos: 196

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 219

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: A, layer: 1, pos: 219

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 95

### Candidate
type: A, layer: 1, pos: 95

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of NS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 215

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1279181, upper bound: 43.1243266
time: 6.74 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1333461, upper bound: 43.1299807
time: 7.83 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
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

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 102

### Candidate
type: B, layer: 1, pos: 102

### Candidate
type: B, layer: 1, pos: 153

### Candidate
type: A, layer: 1, pos: 153

### Candidate
type: A, layer: 1, pos: 161

### Candidate
type: B, layer: 1, pos: 161

### Candidate
type: B, layer: 1, pos: 77

### Candidate
type: A, layer: 1, pos: 77

### Candidate
type: A, layer: 1, pos: 196

### Candidate
type: B, layer: 1, pos: 196

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 219

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: A, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 219

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 95

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 95

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of NS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of NS_A1_B1_A2_A1

### Relational analysis result of NS_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1289488, upper bound: 43.1265785
time: 7.63 seconds

## Relational analysis of NS_A1_B1_A2_A2

### Relational analysis result of NS_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1333461, upper bound: 43.1299808
time: 13.52 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
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

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 102

### Candidate
type: A, layer: 1, pos: 102

### Candidate
type: B, layer: 1, pos: 153

### Candidate
type: A, layer: 1, pos: 153

### Candidate
type: B, layer: 1, pos: 161

### Candidate
type: A, layer: 1, pos: 161

### Candidate
type: A, layer: 1, pos: 77

### Candidate
type: B, layer: 1, pos: 77

### Candidate
type: B, layer: 1, pos: 196

### Candidate
type: A, layer: 1, pos: 196

### Candidate
type: B, layer: 1, pos: 219

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 95

### Candidate
type: A, layer: 1, pos: 219

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 95

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1268875, upper bound: 43.1229023
time: 7.31 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1326738, upper bound: 43.1288669
time: 6.72 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
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

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 102

### Candidate
type: A, layer: 1, pos: 102

### Candidate
type: B, layer: 1, pos: 161

### Candidate
type: A, layer: 1, pos: 161

### Candidate
type: B, layer: 1, pos: 153

### Candidate
type: A, layer: 1, pos: 153

### Candidate
type: B, layer: 1, pos: 77

### Candidate
type: A, layer: 1, pos: 77

### Candidate
type: B, layer: 1, pos: 196

### Candidate
type: A, layer: 1, pos: 196

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: A, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 219

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 219

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 95

### Candidate
type: A, layer: 1, pos: 95

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 215

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1268875, upper bound: 43.1229023
time: 12.07 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1326738, upper bound: 43.1288669
time: 9.48 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
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

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 102

### Candidate
type: B, layer: 1, pos: 153

### Candidate
type: A, layer: 1, pos: 102

### Candidate
type: A, layer: 1, pos: 153

### Candidate
type: B, layer: 1, pos: 77

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 161

### Candidate
type: A, layer: 1, pos: 161

### Candidate
type: A, layer: 1, pos: 77

### Candidate
type: B, layer: 1, pos: 196

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: A, layer: 1, pos: 196

### Candidate
type: B, layer: 1, pos: 219

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1205108, upper bound: 43.1202316
time: 6.09 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1259699, upper bound: 43.1262658
time: 5.39 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -31.5945549, 25.1628647, -22.3406715, 17.8783684, -49.4729233, 47.5035362
1: -28.6641083, 22.5896492, -20.1069565, 16.0682411, -44.7323494, 42.6966057
2: -36.1624641, 22.2232304, -25.3742085, 15.9062481, -52.0687103, 47.5974388
3: -38.8779907, 18.7360210, -27.3010139, 13.6284275, -52.5064163, 46.0370331
4: -36.9562645, 25.6326675, -25.8642807, 18.2503967, -55.2066574, 51.4969482
5: -31.5915909, 24.4638405, -22.2071514, 17.3274460, -48.9190369, 46.6709824
6: -28.6758327, 28.5737953, -20.3151913, 20.2415810, -48.9174080, 48.8889847
7: -32.0679245, 30.6090279, -22.5130692, 21.4846039, -53.5525284, 53.1220970
8: -45.0140991, 20.1968174, -31.7390766, 14.8520393, -59.8661385, 51.9358940
9: -27.9875793, 28.3774071, -19.8298435, 20.1608467, -48.1484261, 48.2072525

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
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
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 36

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 102

### Candidate
type: B, layer: 1, pos: 153

### Candidate
type: A, layer: 1, pos: 102

### Candidate
type: A, layer: 1, pos: 153

### Candidate
type: B, layer: 1, pos: 77

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 161

### Candidate
type: A, layer: 1, pos: 161

### Candidate
type: A, layer: 1, pos: 77

### Candidate
type: B, layer: 1, pos: 196

### Candidate
type: A, layer: 1, pos: 196

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 219

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of NS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of NS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 95

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1205108, upper bound: 43.1202316
time: 7.80 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1259699, upper bound: 43.1262658
time: 5.11 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -28.2203178, 22.4901791, -25.5755291, 20.4365921, -48.6568985, 48.0657043
1: -25.5578384, 20.2087517, -23.0684471, 18.3482170, -43.9060555, 43.2771988
2: -32.2777977, 19.9018517, -29.0784111, 18.1314926, -50.4092865, 48.9802628
3: -34.6678658, 16.7852802, -31.3271523, 15.5016384, -50.1695023, 48.1124306
4: -32.9896088, 22.9196358, -29.6467896, 20.8445110, -53.8341179, 52.5664177
5: -28.1890907, 21.8843651, -25.4572430, 19.7949333, -47.9840202, 47.3416061
6: -25.6103172, 25.5290947, -23.2493610, 23.1563129, -48.7666321, 48.7784576
7: -28.6222458, 27.3599873, -25.8028793, 24.5750523, -53.1972961, 53.1628647
8: -40.3059921, 18.0664539, -36.2212524, 16.9015350, -57.2075272, 54.2877045
9: -24.9904671, 25.3636894, -22.6946526, 23.0477772, -48.0382423, 48.0583420

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 102

### Candidate
type: B, layer: 1, pos: 153

### Candidate
type: A, layer: 1, pos: 102

### Candidate
type: A, layer: 1, pos: 153

### Candidate
type: B, layer: 1, pos: 77

### Candidate
type: B, layer: 1, pos: 161

### Candidate
type: A, layer: 1, pos: 161

### Candidate
type: B, layer: 1, pos: 196

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 77

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 219

### Candidate
type: A, layer: 1, pos: 196

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of NS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of NS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1199827, upper bound: 43.1194178
time: 16.59 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1257516, upper bound: 43.1257516
time: 5.24 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -31.5945549, 25.1628647, -25.5755291, 20.4365921, -52.0311356, 50.7383881
1: -28.6641083, 22.5896492, -23.0684471, 18.3482170, -47.0123253, 45.6580963
2: -36.1624641, 22.2232304, -29.0784111, 18.1314926, -54.2939529, 51.3016434
3: -38.8779907, 18.7360210, -31.3271523, 15.5016384, -54.3796310, 50.0631714
4: -36.9562645, 25.6326675, -29.6467896, 20.8445110, -57.8007660, 55.2794533
5: -31.5915909, 24.4638405, -25.4572430, 19.7949333, -51.3865242, 49.9210815
6: -28.6758327, 28.5737953, -23.2493610, 23.1563129, -51.8321381, 51.8231583
7: -32.0679245, 30.6090279, -25.8028793, 24.5750523, -56.6429749, 56.4119072
8: -45.0140991, 20.1968174, -36.2212524, 16.9015350, -61.9156342, 56.4180679
9: -27.9875793, 28.3774071, -22.6946526, 23.0477772, -51.0353546, 51.0720596

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 36

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 102

### Candidate
type: A, layer: 1, pos: 102

### Candidate
type: B, layer: 1, pos: 153

### Candidate
type: B, layer: 1, pos: 161

### Candidate
type: A, layer: 1, pos: 161

### Candidate
type: B, layer: 1, pos: 77

### Candidate
type: A, layer: 1, pos: 153

### Candidate
type: A, layer: 1, pos: 77

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 196

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: A, layer: 1, pos: 196

### Candidate
type: B, layer: 1, pos: 219

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of NS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of NS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1199827, upper bound: 43.1194178
time: 4.89 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1257516, upper bound: 43.1257516
time: 4.64 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 16.43 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 16.43
Output dim: 8, lower bound: -43.1279181, upper bound: 43.1243266
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 16.43
Output dim: 8, lower bound: -43.1333461, upper bound: 43.1299807
NS_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 16.43
Output dim: 8, lower bound: -43.1289488, upper bound: 43.1265785
NS_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 16.43
Output dim: 8, lower bound: -43.1333461, upper bound: 43.1299808
NS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 16.43
Output dim: 8, lower bound: -43.1268875, upper bound: 43.1229023
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 16.43
Output dim: 8, lower bound: -43.1326738, upper bound: 43.1288669
NS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 16.43
Output dim: 8, lower bound: -43.1268875, upper bound: 43.1229023
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 16.43
Output dim: 8, lower bound: -43.1326738, upper bound: 43.1288669
NS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 16.43
Output dim: 8, lower bound: -43.1205108, upper bound: 43.1202316
NS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 16.43
Output dim: 8, lower bound: -43.1259699, upper bound: 43.1262658
NS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 16.43
Output dim: 8, lower bound: -43.1205108, upper bound: 43.1202316
NS_A2_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 16.43
Output dim: 8, lower bound: -43.1259699, upper bound: 43.1262658
NS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 16.43
Output dim: 8, lower bound: -43.1199827, upper bound: 43.1194178
NS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 16.43
Output dim: 8, lower bound: -43.1257516, upper bound: 43.1257516
NS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 16.43
Output dim: 8, lower bound: -43.1199827, upper bound: 43.1194178
NS_A2_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 16.43
Output dim: 8, lower bound: -43.1257516, upper bound: 43.1257516

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -19.3344955, 15.4992390, -15.3898010, 12.3558903, -31.6903858, 30.8890381
1: -17.4234962, 13.9534588, -13.8285885, 11.1460896, -28.5695801, 27.7820435
2: -21.9741554, 13.7704706, -17.4932499, 10.9727964, -32.9469452, 31.2637215
3: -23.6174583, 11.7354183, -18.7973938, 9.3337946, -32.9512520, 30.5328121
4: -22.4913826, 15.7779474, -17.9422016, 12.5801411, -35.0715141, 33.7201424
5: -19.2608986, 15.0597038, -15.3458433, 12.0617676, -31.3226624, 30.4055481
6: -17.5365734, 17.5540161, -13.9328270, 14.0063486, -31.5429230, 31.4868431
7: -19.4710350, 18.7613297, -15.4504871, 15.0839090, -34.5549393, 34.2118149
8: -27.7970543, 12.6121845, -22.4882469, 9.8468752, -37.6439285, 35.1004333
9: -17.1527615, 17.4686966, -13.6335974, 13.9103622, -31.0631218, 31.1022949

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 36

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 153

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1283195, upper bound: 43.1245443
time: 11.36 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1283195, upper bound: 43.1249578
time: 8.09 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -20.9198322, 16.7540398, -18.8793678, 15.1314487, -36.0512810, 35.6334000
1: -18.8446178, 15.0706091, -17.0034275, 13.6189489, -32.4635658, 32.0740356
2: -23.7708416, 14.9019985, -21.4681511, 13.4595890, -37.2304306, 36.3701477
3: -25.5657730, 12.7449045, -23.0524330, 11.4548540, -37.0206261, 35.7973366
4: -24.2784424, 17.0779095, -21.9582348, 15.4042130, -39.6826553, 39.0361404
5: -20.8112411, 16.2597523, -18.8026390, 14.7184792, -35.5297203, 35.0623856
6: -18.9997845, 18.9756279, -17.1133442, 17.1400280, -36.1398087, 36.0889702
7: -21.0777359, 20.2076378, -19.0004272, 18.3484459, -39.4261780, 39.2080650
8: -29.8882904, 13.7880383, -27.2114067, 12.2907619, -42.1790543, 40.9994431
9: -18.5633278, 18.8852577, -16.7397270, 17.0368118, -35.6001358, 35.6249809

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1301580, upper bound: 43.1275746
time: 6.15 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1301580, upper bound: 43.1308799
time: 53.77 seconds

## BFS NS instance: NS_A1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -18.4051304, 14.7511120, -19.4093723, 15.5602531, -33.9653854, 34.1604843
1: -16.6249886, 13.2905731, -17.4914112, 14.0069637, -30.6319523, 30.7819843
2: -20.9856358, 13.0400715, -22.0589848, 13.8244619, -34.8100929, 35.0990562
3: -22.5689087, 11.0626583, -23.7095451, 11.7843380, -34.3532486, 34.7721977
4: -21.5352039, 14.9921761, -22.5765762, 15.8400326, -37.3752365, 37.5687523
5: -18.4004288, 14.3880615, -19.3349018, 15.1164875, -33.5169106, 33.7229614
6: -16.6637363, 16.7330742, -17.6070595, 17.6222954, -34.2860298, 34.3401260
7: -18.5479660, 18.0238075, -19.5478859, 18.8291340, -37.3770981, 37.5716934
8: -26.7560120, 11.6771135, -27.8954582, 12.6709919, -39.4270020, 39.5725670
9: -16.3050919, 16.6124096, -17.2220669, 17.5388947, -33.8439865, 33.8344765

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 36

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 153

### Candidate
type: A, layer: 1, pos: 153

### Candidate
type: B, layer: 1, pos: 77

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 77

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_A1_B1_A2_A1_B1

### Relational analysis result of NS_A1_B1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1272911, upper bound: 43.1238588
time: 7.22 seconds

## Relational analysis of NS_A1_B1_A2_A1_B2

### Relational analysis result of NS_A1_B1_A2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1272911, upper bound: 43.1265785
time: 16.86 seconds

## BFS NS instance: NS_A1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -21.8682213, 17.5024529, -21.0014610, 16.8198433, -38.6880569, 38.5039101
1: -19.7671318, 15.7421207, -18.9177132, 15.1279221, -34.8950539, 34.6598282
2: -24.9106483, 15.5144444, -23.8627796, 14.9609394, -39.8715782, 39.3772240
3: -26.7919331, 13.1909485, -25.6654415, 12.7976694, -39.5895996, 38.8563881
4: -25.4939957, 17.7971497, -24.3696384, 17.1452103, -42.6392021, 42.1667862
5: -21.8189449, 17.0113869, -20.8917980, 16.3213463, -38.1402817, 37.9031830
6: -19.8211288, 19.8431892, -19.0760307, 19.0497189, -38.8708496, 38.9192200
7: -22.0596809, 21.2379265, -21.1613083, 20.2801285, -42.3398094, 42.3992348
8: -31.4012623, 14.1425285, -29.9945145, 13.8519592, -45.2532196, 44.1370430
9: -19.3891087, 19.7085724, -18.6378460, 18.9607353, -38.3498421, 38.3464203

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_A1_B1_A2_A2_B1

### Relational analysis result of NS_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1279181, upper bound: 43.1243266
time: 5.72 seconds

## Relational analysis of NS_A1_B1_A2_A2_B2

### Relational analysis result of NS_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1279181, upper bound: 43.1299807
time: 16.61 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -20.9198322, 16.7540398, -21.9410686, 17.5617638, -38.4815903, 38.6951027
1: -18.8446178, 15.0706091, -19.8336010, 15.7940311, -34.6386490, 34.9042091
2: -23.7708416, 14.9019985, -24.9928207, 15.5672350, -39.3380737, 39.8948135
3: -25.5657730, 12.7449045, -26.8815479, 13.2385015, -38.8042755, 39.6264496
4: -24.2784424, 17.0779095, -25.5768166, 17.8579273, -42.1363678, 42.6547241
5: -20.8112411, 16.2597523, -21.8911972, 17.0665398, -37.8777809, 38.1509476
6: -18.9997845, 18.9756279, -19.8894539, 19.9099770, -38.9097595, 38.8650780
7: -21.0777359, 20.2076378, -22.1342888, 21.3040810, -42.3818169, 42.3419189
8: -29.8882904, 13.7880383, -31.4967842, 14.1997213, -44.0880127, 45.2848206
9: -18.5633278, 18.8852577, -19.4562950, 19.7770290, -38.3403549, 38.3415527

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1287213, upper bound: 43.1259147
time: 8.36 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1287213, upper bound: 43.1289277
time: 7.66 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -24.0979176, 19.2688541, -21.9410686, 17.5617638, -41.6596718, 41.2099152
1: -21.7685738, 17.3169365, -19.8336010, 15.7940311, -37.5626068, 37.1505356
2: -27.4207726, 17.0887794, -24.9928207, 15.5672350, -42.9880066, 42.0815887
3: -29.5325165, 14.5868034, -26.8815479, 13.2385015, -42.7710190, 41.4683418
4: -28.0103836, 19.6283455, -25.5768166, 17.8579273, -45.8683052, 45.2051620
5: -24.0118580, 18.6866684, -21.8911972, 17.0665398, -41.0783997, 40.5778542
6: -21.8815117, 21.8451767, -19.8894539, 19.9099770, -41.7914886, 41.7346306
7: -24.3205109, 23.2580261, -22.1342888, 21.3040810, -45.6245918, 45.3923149
8: -34.3111267, 15.7891293, -31.4967842, 14.1997213, -48.5108337, 47.2859116
9: -21.3778400, 21.7237663, -19.4562950, 19.7770290, -41.1548691, 41.1800613

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1284987, upper bound: 43.1257789
time: 7.14 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1284987, upper bound: 43.1288669
time: 9.47 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 17.55 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 17.55
Output dim: 8, lower bound: -43.1283195, upper bound: 43.1245443
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 17.55
Output dim: 8, lower bound: -43.1283195, upper bound: 43.1249578
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 17.55
Output dim: 8, lower bound: -43.1301580, upper bound: 43.1275746
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 17.55
Output dim: 8, lower bound: -43.1301580, upper bound: 43.1308799
NS_A1_B1_A2_A1_B1, status: Status.VERIFIED, split count: 5, time: 17.55
Output dim: 8, lower bound: -43.1272911, upper bound: 43.1238588
NS_A1_B1_A2_A1_B2, status: Status.VERIFIED, split count: 5, time: 17.55
Output dim: 8, lower bound: -43.1272911, upper bound: 43.1265785
NS_A1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 17.55
Output dim: 8, lower bound: -43.1279181, upper bound: 43.1243266
NS_A1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 17.55
Output dim: 8, lower bound: -43.1279181, upper bound: 43.1299807
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 17.55
Output dim: 8, lower bound: -43.1287213, upper bound: 43.1259147
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 17.55
Output dim: 8, lower bound: -43.1287213, upper bound: 43.1289277
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 17.55
Output dim: 8, lower bound: -43.1284987, upper bound: 43.1257789
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 17.55
Output dim: 8, lower bound: -43.1284987, upper bound: 43.1288669

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -15.3349409, 12.3111753, -15.3898010, 12.3558903, -27.6908264, 27.7009773
1: -13.7783413, 11.1070042, -13.8285885, 11.1460896, -24.9244308, 24.9355927
2: -17.4315872, 10.9328938, -17.4932499, 10.9727964, -28.4043770, 28.4261436
3: -18.7311211, 9.2989006, -18.7973938, 9.3337946, -28.0649109, 28.0962944
4: -17.8795528, 12.5349941, -17.9422016, 12.5801411, -30.4596939, 30.4771919
5: -15.2913761, 12.0194559, -15.3458433, 12.0617676, -27.3531418, 27.3652992
6: -13.8812923, 13.9566431, -13.9328270, 14.0063486, -27.8876400, 27.8894691
7: -15.3941956, 15.0333881, -15.4504871, 15.0839090, -30.4781036, 30.4838753
8: -22.4156857, 9.8050032, -22.4882469, 9.8468752, -32.2625580, 32.2932510
9: -13.5831318, 13.8578758, -13.6335974, 13.9103622, -27.4934921, 27.4914742

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 153

### Candidate
type: A, layer: 1, pos: 153

### Candidate
type: B, layer: 1, pos: 102

### Candidate
type: A, layer: 1, pos: 102

### Candidate
type: B, layer: 1, pos: 77

### Candidate
type: A, layer: 1, pos: 77

### Candidate
type: B, layer: 1, pos: 161

### Candidate
type: A, layer: 1, pos: 161

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 196

### Candidate
type: A, layer: 1, pos: 196

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: A, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 215

### Candidate
type: A, layer: 1, pos: 215

### Candidate
type: B, layer: 1, pos: 219

### Candidate
type: A, layer: 1, pos: 219

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 95

### Candidate
type: A, layer: 1, pos: 95

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of NS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 117

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of NS_A1_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 112

### Candidate
type: B, layer: 1, pos: 119

## Relational analysis of NS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 119

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1261326, upper bound: 43.1220655
time: 8.49 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1280831, upper bound: 43.1241282
time: 8.10 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -18.8106842, 15.0750828, -15.3898010, 12.3558903, -31.1665745, 30.4648838
1: -16.9401817, 13.5694637, -13.8285885, 11.1460896, -28.0862656, 27.3980522
2: -21.3903122, 13.4097443, -17.4932499, 10.9727964, -32.3631058, 30.9029922
3: -22.9675961, 11.4098463, -18.7973938, 9.3337946, -32.3013916, 30.2072392
4: -21.8795815, 15.3472853, -17.9422016, 12.5801411, -34.4597206, 33.2894821
5: -18.7343979, 14.6659670, -15.3458433, 12.0617676, -30.7961655, 30.0118103
6: -17.0488205, 17.0770302, -13.9328270, 14.0063486, -31.0551682, 31.0098553
7: -18.9299202, 18.2855263, -15.4504871, 15.0839090, -34.0138283, 33.7360153
8: -27.1207428, 12.2375803, -22.4882469, 9.8468752, -36.9676170, 34.7258263
9: -16.6763840, 16.9718609, -13.6335974, 13.9103622, -30.5867462, 30.6054573

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 92

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 153

### Candidate
type: A, layer: 1, pos: 102

### Candidate
type: B, layer: 1, pos: 153

### Candidate
type: A, layer: 1, pos: 77

### Candidate
type: B, layer: 1, pos: 77

### Candidate
type: A, layer: 1, pos: 161

### Candidate
type: B, layer: 1, pos: 161

### Candidate
type: B, layer: 1, pos: 102

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 196

### Candidate
type: A, layer: 1, pos: 215

### Candidate
type: A, layer: 1, pos: 89

### Candidate
type: A, layer: 1, pos: 219

### Candidate
type: B, layer: 1, pos: 196

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 95

### Candidate
type: B, layer: 1, pos: 219

### Candidate
type: B, layer: 1, pos: 215

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A1_B1_A1_B1_A2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1143968, upper bound: 43.1119876
time: 7.72 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1202212, upper bound: 43.1165091
time: 9.54 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -15.3349409, 12.3111753, -18.8793678, 15.1314487, -30.4663887, 31.1905422
1: -13.7783413, 11.1070042, -17.0034275, 13.6189489, -27.3972893, 28.1104298
2: -17.4315872, 10.9328938, -21.4681511, 13.4595890, -30.8911686, 32.4010429
3: -18.7311211, 9.2989006, -23.0524330, 11.4548540, -30.1859684, 32.3513336
4: -17.8795528, 12.5349941, -21.9582348, 15.4042130, -33.2837677, 34.4932289
5: -15.2913761, 12.0194559, -18.8026390, 14.7184792, -30.0098553, 30.8220940
6: -13.8812923, 13.9566431, -17.1133442, 17.1400280, -31.0213089, 31.0699863
7: -15.3941956, 15.0333881, -19.0004272, 18.3484459, -33.7426376, 34.0338135
8: -22.4156857, 9.8050032, -27.2114067, 12.2907619, -34.7064476, 37.0164108
9: -13.5831318, 13.8578758, -16.7397270, 17.0368118, -30.6199360, 30.5976028

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 92

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 153

### Candidate
type: B, layer: 1, pos: 102

### Candidate
type: A, layer: 1, pos: 153

### Candidate
type: B, layer: 1, pos: 77

### Candidate
type: A, layer: 1, pos: 77

### Candidate
type: B, layer: 1, pos: 161

### Candidate
type: A, layer: 1, pos: 161

### Candidate
type: A, layer: 1, pos: 102

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 196

### Candidate
type: B, layer: 1, pos: 215

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 219

### Candidate
type: A, layer: 1, pos: 196

### Candidate
type: A, layer: 1, pos: 89

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 95

### Candidate
type: A, layer: 1, pos: 219

### Candidate
type: A, layer: 1, pos: 215

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1261326, upper bound: 43.1252775
time: 8.11 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1280831, upper bound: 43.1269615
time: 10.23 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -18.8106842, 15.0750828, -18.8793678, 15.1314487, -33.9421272, 33.9544449
1: -16.9401817, 13.5694637, -17.0034275, 13.6189489, -30.5591316, 30.5728912
2: -21.3903122, 13.4097443, -21.4681511, 13.4595890, -34.8498993, 34.8778915
3: -22.9675961, 11.4098463, -23.0524330, 11.4548540, -34.4224510, 34.4622803
4: -21.8795815, 15.3472853, -21.9582348, 15.4042130, -37.2837944, 37.3055191
5: -18.7343979, 14.6659670, -18.8026390, 14.7184792, -33.4528770, 33.4686050
6: -17.0488205, 17.0770302, -17.1133442, 17.1400280, -34.1888428, 34.1903725
7: -18.9299202, 18.2855263, -19.0004272, 18.3484459, -37.2783661, 37.2859535
8: -27.1207428, 12.2375803, -27.2114067, 12.2907619, -39.4115067, 39.4489861
9: -16.6763840, 16.9718609, -16.7397270, 17.0368118, -33.7131958, 33.7115822

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
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
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 102

### Candidate
type: A, layer: 1, pos: 102

### Candidate
type: B, layer: 1, pos: 153

### Candidate
type: A, layer: 1, pos: 153

### Candidate
type: B, layer: 1, pos: 161

### Candidate
type: A, layer: 1, pos: 161

### Candidate
type: B, layer: 1, pos: 77

### Candidate
type: A, layer: 1, pos: 77

### Candidate
type: B, layer: 1, pos: 196

### Candidate
type: A, layer: 1, pos: 196

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: A, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 219

### Candidate
type: A, layer: 1, pos: 219

### Candidate
type: B, layer: 1, pos: 215

### Candidate
type: A, layer: 1, pos: 215

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 13.39 + 586.75 = 600.14 seconds
