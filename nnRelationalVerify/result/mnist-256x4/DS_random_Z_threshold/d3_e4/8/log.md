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
execution time: IAR + RelationalAnalysis = 0.87 + 11.95 = 12.83 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -43.1707437, upper bound: 43.1707437

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 103

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1388014, upper bound: 43.1388014
time: 6.20 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1388014, upper bound: 43.1388014
time: 6.57 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 12.78 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 12.78
Output dim: 8, lower bound: -43.1388014, upper bound: 43.1388014
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 12.78
Output dim: 8, lower bound: -43.1388014, upper bound: 43.1388014

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -24.6925583, 19.7640648, -24.6925583, 19.7640648, -44.4566231, 44.4566231
1: -22.1655941, 17.7110901, -22.1655941, 17.7110901, -39.8766861, 39.8766861
2: -28.0165939, 17.5855999, -28.0165939, 17.5855999, -45.6021957, 45.6021957
3: -30.1115532, 15.0882940, -30.1115532, 15.0882940, -45.1998444, 45.1998482
4: -28.4748173, 20.2006111, -28.4748173, 20.2006111, -48.6754303, 48.6754303
5: -24.4868717, 19.1075554, -24.4868717, 19.1075554, -43.5944290, 43.5944290
6: -22.5470924, 22.3522205, -22.5470924, 22.3522205, -44.8993111, 44.8993149
7: -24.8867416, 23.5568867, -24.8867416, 23.5568867, -48.4436264, 48.4436264
8: -34.8141861, 16.7162991, -34.8141861, 16.7162991, -51.5304832, 51.5304871
9: -21.9554176, 22.3245659, -21.9554176, 22.3245659, -44.2799835, 44.2799835

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1364204, upper bound: 43.1364257
time: 17.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1364257, upper bound: 43.1364204
time: 9.91 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -24.6925583, 19.7640648, -24.6925583, 19.7640648, -44.4566231, 44.4566231
1: -22.1655941, 17.7110901, -22.1655941, 17.7110901, -39.8766861, 39.8766861
2: -28.0165939, 17.5855999, -28.0165939, 17.5855999, -45.6021957, 45.6021957
3: -30.1115532, 15.0882940, -30.1115532, 15.0882940, -45.1998444, 45.1998482
4: -28.4748173, 20.2006111, -28.4748173, 20.2006111, -48.6754303, 48.6754303
5: -24.4868717, 19.1075554, -24.4868717, 19.1075554, -43.5944290, 43.5944290
6: -22.5470924, 22.3522205, -22.5470924, 22.3522205, -44.8993111, 44.8993149
7: -24.8867416, 23.5568867, -24.8867416, 23.5568867, -48.4436264, 48.4436264
8: -34.8141861, 16.7162991, -34.8141861, 16.7162991, -51.5304832, 51.5304871
9: -21.9554176, 22.3245659, -21.9554176, 22.3245659, -44.2799835, 44.2799835

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 251

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1304251, upper bound: 43.1304251
time: 6.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1304251, upper bound: 43.1304251
time: 8.24 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 19.32 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 19.32
Output dim: 8, lower bound: -43.1364204, upper bound: 43.1364257
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 19.32
Output dim: 8, lower bound: -43.1364257, upper bound: 43.1364204
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 19.32
Output dim: 8, lower bound: -43.1304251, upper bound: 43.1304251
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 19.32
Output dim: 8, lower bound: -43.1304251, upper bound: 43.1304251

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -24.6925583, 19.7640648, -24.6925583, 19.7640648, -44.4566231, 44.4566231
1: -22.1655941, 17.7110901, -22.1655941, 17.7110901, -39.8766861, 39.8766861
2: -28.0165939, 17.5855999, -28.0165939, 17.5855999, -45.6021957, 45.6021957
3: -30.1115532, 15.0882940, -30.1115532, 15.0882940, -45.1998444, 45.1998482
4: -28.4748173, 20.2006111, -28.4748173, 20.2006111, -48.6754303, 48.6754303
5: -24.4868717, 19.1075554, -24.4868717, 19.1075554, -43.5944290, 43.5944290
6: -22.5470924, 22.3522205, -22.5470924, 22.3522205, -44.8993111, 44.8993149
7: -24.8867416, 23.5568867, -24.8867416, 23.5568867, -48.4436264, 48.4436264
8: -34.8141861, 16.7162991, -34.8141861, 16.7162991, -51.5304832, 51.5304871
9: -21.9554176, 22.3245659, -21.9554176, 22.3245659, -44.2799835, 44.2799835

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 92

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1364166, upper bound: 43.1364257
time: 4.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1364204, upper bound: 43.1364209
time: 11.06 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -24.6925583, 19.7640648, -24.6925583, 19.7640648, -44.4566231, 44.4566231
1: -22.1655941, 17.7110901, -22.1655941, 17.7110901, -39.8766861, 39.8766861
2: -28.0165939, 17.5855999, -28.0165939, 17.5855999, -45.6021957, 45.6021957
3: -30.1115532, 15.0882940, -30.1115532, 15.0882940, -45.1998444, 45.1998482
4: -28.4748173, 20.2006111, -28.4748173, 20.2006111, -48.6754303, 48.6754303
5: -24.4868717, 19.1075554, -24.4868717, 19.1075554, -43.5944290, 43.5944290
6: -22.5470924, 22.3522205, -22.5470924, 22.3522205, -44.8993111, 44.8993149
7: -24.8867416, 23.5568867, -24.8867416, 23.5568867, -48.4436264, 48.4436264
8: -34.8141861, 16.7162991, -34.8141861, 16.7162991, -51.5304832, 51.5304871
9: -21.9554176, 22.3245659, -21.9554176, 22.3245659, -44.2799835, 44.2799835

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1364192, upper bound: 43.1364204
time: 5.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1364257, upper bound: 43.1364167
time: 5.81 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -24.6925583, 19.7640648, -24.6925583, 19.7640648, -44.4566231, 44.4566231
1: -22.1655941, 17.7110901, -22.1655941, 17.7110901, -39.8766861, 39.8766861
2: -28.0165939, 17.5855999, -28.0165939, 17.5855999, -45.6021957, 45.6021957
3: -30.1115532, 15.0882940, -30.1115532, 15.0882940, -45.1998444, 45.1998482
4: -28.4748173, 20.2006111, -28.4748173, 20.2006111, -48.6754303, 48.6754303
5: -24.4868717, 19.1075554, -24.4868717, 19.1075554, -43.5944290, 43.5944290
6: -22.5470924, 22.3522205, -22.5470924, 22.3522205, -44.8993111, 44.8993149
7: -24.8867416, 23.5568867, -24.8867416, 23.5568867, -48.4436264, 48.4436264
8: -34.8141861, 16.7162991, -34.8141861, 16.7162991, -51.5304832, 51.5304871
9: -21.9554176, 22.3245659, -21.9554176, 22.3245659, -44.2799835, 44.2799835

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1246517, upper bound: 43.1246517
time: 6.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1246517, upper bound: 43.1246517
time: 5.40 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -24.6925583, 19.7640648, -24.6925583, 19.7640648, -44.4566231, 44.4566231
1: -22.1655941, 17.7110901, -22.1655941, 17.7110901, -39.8766861, 39.8766861
2: -28.0165939, 17.5855999, -28.0165939, 17.5855999, -45.6021957, 45.6021957
3: -30.1115532, 15.0882940, -30.1115532, 15.0882940, -45.1998444, 45.1998482
4: -28.4748173, 20.2006111, -28.4748173, 20.2006111, -48.6754303, 48.6754303
5: -24.4868717, 19.1075554, -24.4868717, 19.1075554, -43.5944290, 43.5944290
6: -22.5470924, 22.3522205, -22.5470924, 22.3522205, -44.8993111, 44.8993149
7: -24.8867416, 23.5568867, -24.8867416, 23.5568867, -48.4436264, 48.4436264
8: -34.8141861, 16.7162991, -34.8141861, 16.7162991, -51.5304832, 51.5304871
9: -21.9554176, 22.3245659, -21.9554176, 22.3245659, -44.2799835, 44.2799835

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1297481, upper bound: 43.1297484
time: 5.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1297484, upper bound: 43.1297481
time: 5.23 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 11.53 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 11.53
Output dim: 8, lower bound: -43.1364166, upper bound: 43.1364257
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 11.53
Output dim: 8, lower bound: -43.1364204, upper bound: 43.1364209
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 11.53
Output dim: 8, lower bound: -43.1364192, upper bound: 43.1364204
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 11.53
Output dim: 8, lower bound: -43.1364257, upper bound: 43.1364167
DS_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 11.53
Output dim: 8, lower bound: -43.1246517, upper bound: 43.1246517
DS_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 11.53
Output dim: 8, lower bound: -43.1246517, upper bound: 43.1246517
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 11.53
Output dim: 8, lower bound: -43.1297481, upper bound: 43.1297484
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 11.53
Output dim: 8, lower bound: -43.1297484, upper bound: 43.1297481

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -24.6925583, 19.7640648, -24.6925583, 19.7640648, -44.4566231, 44.4566231
1: -22.1655941, 17.7110901, -22.1655941, 17.7110901, -39.8766861, 39.8766861
2: -28.0165939, 17.5855999, -28.0165939, 17.5855999, -45.6021957, 45.6021957
3: -30.1115532, 15.0882940, -30.1115532, 15.0882940, -45.1998444, 45.1998482
4: -28.4748173, 20.2006111, -28.4748173, 20.2006111, -48.6754303, 48.6754303
5: -24.4868717, 19.1075554, -24.4868717, 19.1075554, -43.5944290, 43.5944290
6: -22.5470924, 22.3522205, -22.5470924, 22.3522205, -44.8993111, 44.8993149
7: -24.8867416, 23.5568867, -24.8867416, 23.5568867, -48.4436264, 48.4436264
8: -34.8141861, 16.7162991, -34.8141861, 16.7162991, -51.5304832, 51.5304871
9: -21.9554176, 22.3245659, -21.9554176, 22.3245659, -44.2799835, 44.2799835

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1318122, upper bound: 43.1318140
time: 5.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1318122, upper bound: 43.1318140
time: 5.10 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -24.6925583, 19.7640648, -24.6925583, 19.7640648, -44.4566231, 44.4566231
1: -22.1655941, 17.7110901, -22.1655941, 17.7110901, -39.8766861, 39.8766861
2: -28.0165939, 17.5855999, -28.0165939, 17.5855999, -45.6021957, 45.6021957
3: -30.1115532, 15.0882940, -30.1115532, 15.0882940, -45.1998444, 45.1998482
4: -28.4748173, 20.2006111, -28.4748173, 20.2006111, -48.6754303, 48.6754303
5: -24.4868717, 19.1075554, -24.4868717, 19.1075554, -43.5944290, 43.5944290
6: -22.5470924, 22.3522205, -22.5470924, 22.3522205, -44.8993111, 44.8993149
7: -24.8867416, 23.5568867, -24.8867416, 23.5568867, -48.4436264, 48.4436264
8: -34.8141861, 16.7162991, -34.8141861, 16.7162991, -51.5304832, 51.5304871
9: -21.9554176, 22.3245659, -21.9554176, 22.3245659, -44.2799835, 44.2799835

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 219

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 95

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 117

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 183

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 197

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 216

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1360077, upper bound: 43.1360079
time: 6.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1360077, upper bound: 43.1360079
time: 6.74 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -24.6925583, 19.7640648, -24.6925583, 19.7640648, -44.4566231, 44.4566231
1: -22.1655941, 17.7110901, -22.1655941, 17.7110901, -39.8766861, 39.8766861
2: -28.0165939, 17.5855999, -28.0165939, 17.5855999, -45.6021957, 45.6021957
3: -30.1115532, 15.0882940, -30.1115532, 15.0882940, -45.1998444, 45.1998482
4: -28.4748173, 20.2006111, -28.4748173, 20.2006111, -48.6754303, 48.6754303
5: -24.4868717, 19.1075554, -24.4868717, 19.1075554, -43.5944290, 43.5944290
6: -22.5470924, 22.3522205, -22.5470924, 22.3522205, -44.8993111, 44.8993149
7: -24.8867416, 23.5568867, -24.8867416, 23.5568867, -48.4436264, 48.4436264
8: -34.8141861, 16.7162991, -34.8141861, 16.7162991, -51.5304832, 51.5304871
9: -21.9554176, 22.3245659, -21.9554176, 22.3245659, -44.2799835, 44.2799835

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 251

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1274836, upper bound: 43.1274831
time: 5.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1274836, upper bound: 43.1274831
time: 5.81 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -24.6925583, 19.7640648, -24.6925583, 19.7640648, -44.4566231, 44.4566231
1: -22.1655941, 17.7110901, -22.1655941, 17.7110901, -39.8766861, 39.8766861
2: -28.0165939, 17.5855999, -28.0165939, 17.5855999, -45.6021957, 45.6021957
3: -30.1115532, 15.0882940, -30.1115532, 15.0882940, -45.1998444, 45.1998482
4: -28.4748173, 20.2006111, -28.4748173, 20.2006111, -48.6754303, 48.6754303
5: -24.4868717, 19.1075554, -24.4868717, 19.1075554, -43.5944290, 43.5944290
6: -22.5470924, 22.3522205, -22.5470924, 22.3522205, -44.8993111, 44.8993149
7: -24.8867416, 23.5568867, -24.8867416, 23.5568867, -48.4436264, 48.4436264
8: -34.8141861, 16.7162991, -34.8141861, 16.7162991, -51.5304832, 51.5304871
9: -21.9554176, 22.3245659, -21.9554176, 22.3245659, -44.2799835, 44.2799835

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 103

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 77

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 183

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 214

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 251

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1274844, upper bound: 43.1274819
time: 5.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1274844, upper bound: 43.1274819
time: 4.97 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -24.6925583, 19.7640648, -24.6925583, 19.7640648, -44.4566231, 44.4566231
1: -22.1655941, 17.7110901, -22.1655941, 17.7110901, -39.8766861, 39.8766861
2: -28.0165939, 17.5855999, -28.0165939, 17.5855999, -45.6021957, 45.6021957
3: -30.1115532, 15.0882940, -30.1115532, 15.0882940, -45.1998444, 45.1998482
4: -28.4748173, 20.2006111, -28.4748173, 20.2006111, -48.6754303, 48.6754303
5: -24.4868717, 19.1075554, -24.4868717, 19.1075554, -43.5944290, 43.5944290
6: -22.5470924, 22.3522205, -22.5470924, 22.3522205, -44.8993111, 44.8993149
7: -24.8867416, 23.5568867, -24.8867416, 23.5568867, -48.4436264, 48.4436264
8: -34.8141861, 16.7162991, -34.8141861, 16.7162991, -51.5304832, 51.5304871
9: -21.9554176, 22.3245659, -21.9554176, 22.3245659, -44.2799835, 44.2799835

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1297456, upper bound: 43.1297484
time: 5.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1297481, upper bound: 43.1297470
time: 4.49 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -24.6925583, 19.7640648, -24.6925583, 19.7640648, -44.4566231, 44.4566231
1: -22.1655941, 17.7110901, -22.1655941, 17.7110901, -39.8766861, 39.8766861
2: -28.0165939, 17.5855999, -28.0165939, 17.5855999, -45.6021957, 45.6021957
3: -30.1115532, 15.0882940, -30.1115532, 15.0882940, -45.1998444, 45.1998482
4: -28.4748173, 20.2006111, -28.4748173, 20.2006111, -48.6754303, 48.6754303
5: -24.4868717, 19.1075554, -24.4868717, 19.1075554, -43.5944290, 43.5944290
6: -22.5470924, 22.3522205, -22.5470924, 22.3522205, -44.8993111, 44.8993149
7: -24.8867416, 23.5568867, -24.8867416, 23.5568867, -48.4436264, 48.4436264
8: -34.8141861, 16.7162991, -34.8141861, 16.7162991, -51.5304832, 51.5304871
9: -21.9554176, 22.3245659, -21.9554176, 22.3245659, -44.2799835, 44.2799835

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 219

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 153

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1148429, upper bound: 43.1148428
time: 6.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1148429, upper bound: 43.1148428
time: 6.15 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 19.20 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 19.20
Output dim: 8, lower bound: -43.1318122, upper bound: 43.1318140
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 19.20
Output dim: 8, lower bound: -43.1318122, upper bound: 43.1318140
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 19.20
Output dim: 8, lower bound: -43.1360077, upper bound: 43.1360079
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 19.20
Output dim: 8, lower bound: -43.1360077, upper bound: 43.1360079
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 19.20
Output dim: 8, lower bound: -43.1274836, upper bound: 43.1274831
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 19.20
Output dim: 8, lower bound: -43.1274836, upper bound: 43.1274831
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 19.20
Output dim: 8, lower bound: -43.1274844, upper bound: 43.1274819
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 19.20
Output dim: 8, lower bound: -43.1274844, upper bound: 43.1274819
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 19.20
Output dim: 8, lower bound: -43.1297456, upper bound: 43.1297484
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 19.20
Output dim: 8, lower bound: -43.1297481, upper bound: 43.1297470
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 19.20
Output dim: 8, lower bound: -43.1148429, upper bound: 43.1148428
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 19.20
Output dim: 8, lower bound: -43.1148429, upper bound: 43.1148428

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -24.6925583, 19.7640648, -24.6925583, 19.7640648, -44.4566231, 44.4566231
1: -22.1655941, 17.7110901, -22.1655941, 17.7110901, -39.8766861, 39.8766861
2: -28.0165939, 17.5855999, -28.0165939, 17.5855999, -45.6021957, 45.6021957
3: -30.1115532, 15.0882940, -30.1115532, 15.0882940, -45.1998444, 45.1998482
4: -28.4748173, 20.2006111, -28.4748173, 20.2006111, -48.6754303, 48.6754303
5: -24.4868717, 19.1075554, -24.4868717, 19.1075554, -43.5944290, 43.5944290
6: -22.5470924, 22.3522205, -22.5470924, 22.3522205, -44.8993111, 44.8993149
7: -24.8867416, 23.5568867, -24.8867416, 23.5568867, -48.4436264, 48.4436264
8: -34.8141861, 16.7162991, -34.8141861, 16.7162991, -51.5304832, 51.5304871
9: -21.9554176, 22.3245659, -21.9554176, 22.3245659, -44.2799835, 44.2799835

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1297255, upper bound: 43.1297283
time: 5.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1297258, upper bound: 43.1297282
time: 5.05 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -24.6925583, 19.7640648, -24.6925583, 19.7640648, -44.4566231, 44.4566231
1: -22.1655941, 17.7110901, -22.1655941, 17.7110901, -39.8766861, 39.8766861
2: -28.0165939, 17.5855999, -28.0165939, 17.5855999, -45.6021957, 45.6021957
3: -30.1115532, 15.0882940, -30.1115532, 15.0882940, -45.1998444, 45.1998482
4: -28.4748173, 20.2006111, -28.4748173, 20.2006111, -48.6754303, 48.6754303
5: -24.4868717, 19.1075554, -24.4868717, 19.1075554, -43.5944290, 43.5944290
6: -22.5470924, 22.3522205, -22.5470924, 22.3522205, -44.8993111, 44.8993149
7: -24.8867416, 23.5568867, -24.8867416, 23.5568867, -48.4436264, 48.4436264
8: -34.8141861, 16.7162991, -34.8141861, 16.7162991, -51.5304832, 51.5304871
9: -21.9554176, 22.3245659, -21.9554176, 22.3245659, -44.2799835, 44.2799835

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 52

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1318118, upper bound: 43.1318140
time: 5.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1318122, upper bound: 43.1318131
time: 6.52 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -24.6925583, 19.7640648, -24.6925583, 19.7640648, -44.4566231, 44.4566231
1: -22.1655941, 17.7110901, -22.1655941, 17.7110901, -39.8766861, 39.8766861
2: -28.0165939, 17.5855999, -28.0165939, 17.5855999, -45.6021957, 45.6021957
3: -30.1115532, 15.0882940, -30.1115532, 15.0882940, -45.1998444, 45.1998482
4: -28.4748173, 20.2006111, -28.4748173, 20.2006111, -48.6754303, 48.6754303
5: -24.4868717, 19.1075554, -24.4868717, 19.1075554, -43.5944290, 43.5944290
6: -22.5470924, 22.3522205, -22.5470924, 22.3522205, -44.8993111, 44.8993149
7: -24.8867416, 23.5568867, -24.8867416, 23.5568867, -48.4436264, 48.4436264
8: -34.8141861, 16.7162991, -34.8141861, 16.7162991, -51.5304832, 51.5304871
9: -21.9554176, 22.3245659, -21.9554176, 22.3245659, -44.2799835, 44.2799835

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1360050, upper bound: 43.1360079
time: 7.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1360077, upper bound: 43.1360050
time: 10.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -24.6925583, 19.7640648, -24.6925583, 19.7640648, -44.4566231, 44.4566231
1: -22.1655941, 17.7110901, -22.1655941, 17.7110901, -39.8766861, 39.8766861
2: -28.0165939, 17.5855999, -28.0165939, 17.5855999, -45.6021957, 45.6021957
3: -30.1115532, 15.0882940, -30.1115532, 15.0882940, -45.1998444, 45.1998482
4: -28.4748173, 20.2006111, -28.4748173, 20.2006111, -48.6754303, 48.6754303
5: -24.4868717, 19.1075554, -24.4868717, 19.1075554, -43.5944290, 43.5944290
6: -22.5470924, 22.3522205, -22.5470924, 22.3522205, -44.8993111, 44.8993149
7: -24.8867416, 23.5568867, -24.8867416, 23.5568867, -48.4436264, 48.4436264
8: -34.8141861, 16.7162991, -34.8141861, 16.7162991, -51.5304832, 51.5304871
9: -21.9554176, 22.3245659, -21.9554176, 22.3245659, -44.2799835, 44.2799835

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 215

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 214

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1360050, upper bound: 43.1360079
time: 5.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1360077, upper bound: 43.1360050
time: 5.35 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -24.6925583, 19.7640648, -24.6925583, 19.7640648, -44.4566231, 44.4566231
1: -22.1655941, 17.7110901, -22.1655941, 17.7110901, -39.8766861, 39.8766861
2: -28.0165939, 17.5855999, -28.0165939, 17.5855999, -45.6021957, 45.6021957
3: -30.1115532, 15.0882940, -30.1115532, 15.0882940, -45.1998444, 45.1998482
4: -28.4748173, 20.2006111, -28.4748173, 20.2006111, -48.6754303, 48.6754303
5: -24.4868717, 19.1075554, -24.4868717, 19.1075554, -43.5944290, 43.5944290
6: -22.5470924, 22.3522205, -22.5470924, 22.3522205, -44.8993111, 44.8993149
7: -24.8867416, 23.5568867, -24.8867416, 23.5568867, -48.4436264, 48.4436264
8: -34.8141861, 16.7162991, -34.8141861, 16.7162991, -51.5304832, 51.5304871
9: -21.9554176, 22.3245659, -21.9554176, 22.3245659, -44.2799835, 44.2799835

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 117

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1297430, upper bound: 43.1297484
time: 6.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1297456, upper bound: 43.1297461
time: 6.57 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -24.6925583, 19.7640648, -24.6925583, 19.7640648, -44.4566231, 44.4566231
1: -22.1655941, 17.7110901, -22.1655941, 17.7110901, -39.8766861, 39.8766861
2: -28.0165939, 17.5855999, -28.0165939, 17.5855999, -45.6021957, 45.6021957
3: -30.1115532, 15.0882940, -30.1115532, 15.0882940, -45.1998444, 45.1998482
4: -28.4748173, 20.2006111, -28.4748173, 20.2006111, -48.6754303, 48.6754303
5: -24.4868717, 19.1075554, -24.4868717, 19.1075554, -43.5944290, 43.5944290
6: -22.5470924, 22.3522205, -22.5470924, 22.3522205, -44.8993111, 44.8993149
7: -24.8867416, 23.5568867, -24.8867416, 23.5568867, -48.4436264, 48.4436264
8: -34.8141861, 16.7162991, -34.8141861, 16.7162991, -51.5304832, 51.5304871
9: -21.9554176, 22.3245659, -21.9554176, 22.3245659, -44.2799835, 44.2799835

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 52

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1297481, upper bound: 43.1297390
time: 5.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1297446, upper bound: 43.1297470
time: 3.80 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 10.14 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.14
Output dim: 8, lower bound: -43.1297255, upper bound: 43.1297283
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.14
Output dim: 8, lower bound: -43.1297258, upper bound: 43.1297282
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.14
Output dim: 8, lower bound: -43.1318118, upper bound: 43.1318140
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.14
Output dim: 8, lower bound: -43.1318122, upper bound: 43.1318131
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.14
Output dim: 8, lower bound: -43.1360050, upper bound: 43.1360079
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.14
Output dim: 8, lower bound: -43.1360077, upper bound: 43.1360050
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.14
Output dim: 8, lower bound: -43.1360050, upper bound: 43.1360079
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.14
Output dim: 8, lower bound: -43.1360077, upper bound: 43.1360050
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.14
Output dim: 8, lower bound: -43.1297430, upper bound: 43.1297484
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.14
Output dim: 8, lower bound: -43.1297456, upper bound: 43.1297461
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.14
Output dim: 8, lower bound: -43.1297481, upper bound: 43.1297390
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.14
Output dim: 8, lower bound: -43.1297446, upper bound: 43.1297470

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -24.6925583, 19.7640648, -24.6925583, 19.7640648, -44.4566231, 44.4566231
1: -22.1655941, 17.7110901, -22.1655941, 17.7110901, -39.8766861, 39.8766861
2: -28.0165939, 17.5855999, -28.0165939, 17.5855999, -45.6021957, 45.6021957
3: -30.1115532, 15.0882940, -30.1115532, 15.0882940, -45.1998444, 45.1998482
4: -28.4748173, 20.2006111, -28.4748173, 20.2006111, -48.6754303, 48.6754303
5: -24.4868717, 19.1075554, -24.4868717, 19.1075554, -43.5944290, 43.5944290
6: -22.5470924, 22.3522205, -22.5470924, 22.3522205, -44.8993111, 44.8993149
7: -24.8867416, 23.5568867, -24.8867416, 23.5568867, -48.4436264, 48.4436264
8: -34.8141861, 16.7162991, -34.8141861, 16.7162991, -51.5304832, 51.5304871
9: -21.9554176, 22.3245659, -21.9554176, 22.3245659, -44.2799835, 44.2799835

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 215

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 112

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1297247, upper bound: 43.1297283
time: 6.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1297255, upper bound: 43.1297273
time: 6.09 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -24.6925583, 19.7640648, -24.6925583, 19.7640648, -44.4566231, 44.4566231
1: -22.1655941, 17.7110901, -22.1655941, 17.7110901, -39.8766861, 39.8766861
2: -28.0165939, 17.5855999, -28.0165939, 17.5855999, -45.6021957, 45.6021957
3: -30.1115532, 15.0882940, -30.1115532, 15.0882940, -45.1998444, 45.1998482
4: -28.4748173, 20.2006111, -28.4748173, 20.2006111, -48.6754303, 48.6754303
5: -24.4868717, 19.1075554, -24.4868717, 19.1075554, -43.5944290, 43.5944290
6: -22.5470924, 22.3522205, -22.5470924, 22.3522205, -44.8993111, 44.8993149
7: -24.8867416, 23.5568867, -24.8867416, 23.5568867, -48.4436264, 48.4436264
8: -34.8141861, 16.7162991, -34.8141861, 16.7162991, -51.5304832, 51.5304871
9: -21.9554176, 22.3245659, -21.9554176, 22.3245659, -44.2799835, 44.2799835

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 95

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 215

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 127

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1297258, upper bound: 43.1297275
time: 5.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1297242, upper bound: 43.1297282
time: 15.09 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -24.6925583, 19.7640648, -24.6925583, 19.7640648, -44.4566231, 44.4566231
1: -22.1655941, 17.7110901, -22.1655941, 17.7110901, -39.8766861, 39.8766861
2: -28.0165939, 17.5855999, -28.0165939, 17.5855999, -45.6021957, 45.6021957
3: -30.1115532, 15.0882940, -30.1115532, 15.0882940, -45.1998444, 45.1998482
4: -28.4748173, 20.2006111, -28.4748173, 20.2006111, -48.6754303, 48.6754303
5: -24.4868717, 19.1075554, -24.4868717, 19.1075554, -43.5944290, 43.5944290
6: -22.5470924, 22.3522205, -22.5470924, 22.3522205, -44.8993111, 44.8993149
7: -24.8867416, 23.5568867, -24.8867416, 23.5568867, -48.4436264, 48.4436264
8: -34.8141861, 16.7162991, -34.8141861, 16.7162991, -51.5304832, 51.5304871
9: -21.9554176, 22.3245659, -21.9554176, 22.3245659, -44.2799835, 44.2799835

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 77

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 197

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1318118, upper bound: 43.1318140
time: 6.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1318118, upper bound: 43.1318133
time: 5.91 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -24.6925583, 19.7640648, -24.6925583, 19.7640648, -44.4566231, 44.4566231
1: -22.1655941, 17.7110901, -22.1655941, 17.7110901, -39.8766861, 39.8766861
2: -28.0165939, 17.5855999, -28.0165939, 17.5855999, -45.6021957, 45.6021957
3: -30.1115532, 15.0882940, -30.1115532, 15.0882940, -45.1998444, 45.1998482
4: -28.4748173, 20.2006111, -28.4748173, 20.2006111, -48.6754303, 48.6754303
5: -24.4868717, 19.1075554, -24.4868717, 19.1075554, -43.5944290, 43.5944290
6: -22.5470924, 22.3522205, -22.5470924, 22.3522205, -44.8993111, 44.8993149
7: -24.8867416, 23.5568867, -24.8867416, 23.5568867, -48.4436264, 48.4436264
8: -34.8141861, 16.7162991, -34.8141861, 16.7162991, -51.5304832, 51.5304871
9: -21.9554176, 22.3245659, -21.9554176, 22.3245659, -44.2799835, 44.2799835

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 127

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1317924, upper bound: 43.1317886
time: 4.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1317864, upper bound: 43.1317935
time: 6.02 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -24.6925583, 19.7640648, -24.6925583, 19.7640648, -44.4566231, 44.4566231
1: -22.1655941, 17.7110901, -22.1655941, 17.7110901, -39.8766861, 39.8766861
2: -28.0165939, 17.5855999, -28.0165939, 17.5855999, -45.6021957, 45.6021957
3: -30.1115532, 15.0882940, -30.1115532, 15.0882940, -45.1998444, 45.1998482
4: -28.4748173, 20.2006111, -28.4748173, 20.2006111, -48.6754303, 48.6754303
5: -24.4868717, 19.1075554, -24.4868717, 19.1075554, -43.5944290, 43.5944290
6: -22.5470924, 22.3522205, -22.5470924, 22.3522205, -44.8993111, 44.8993149
7: -24.8867416, 23.5568867, -24.8867416, 23.5568867, -48.4436264, 48.4436264
8: -34.8141861, 16.7162991, -34.8141861, 16.7162991, -51.5304832, 51.5304871
9: -21.9554176, 22.3245659, -21.9554176, 22.3245659, -44.2799835, 44.2799835

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 215

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 52

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1360050, upper bound: 43.1360079
time: 5.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1360049, upper bound: 43.1360069
time: 5.42 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -24.6925583, 19.7640648, -24.6925583, 19.7640648, -44.4566231, 44.4566231
1: -22.1655941, 17.7110901, -22.1655941, 17.7110901, -39.8766861, 39.8766861
2: -28.0165939, 17.5855999, -28.0165939, 17.5855999, -45.6021957, 45.6021957
3: -30.1115532, 15.0882940, -30.1115532, 15.0882940, -45.1998444, 45.1998482
4: -28.4748173, 20.2006111, -28.4748173, 20.2006111, -48.6754303, 48.6754303
5: -24.4868717, 19.1075554, -24.4868717, 19.1075554, -43.5944290, 43.5944290
6: -22.5470924, 22.3522205, -22.5470924, 22.3522205, -44.8993111, 44.8993149
7: -24.8867416, 23.5568867, -24.8867416, 23.5568867, -48.4436264, 48.4436264
8: -34.8141861, 16.7162991, -34.8141861, 16.7162991, -51.5304832, 51.5304871
9: -21.9554176, 22.3245659, -21.9554176, 22.3245659, -44.2799835, 44.2799835

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 215

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1341998, upper bound: 43.1342012
time: 13.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1341998, upper bound: 43.1342012
time: 15.33 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -24.6925583, 19.7640648, -24.6925583, 19.7640648, -44.4566231, 44.4566231
1: -22.1655941, 17.7110901, -22.1655941, 17.7110901, -39.8766861, 39.8766861
2: -28.0165939, 17.5855999, -28.0165939, 17.5855999, -45.6021957, 45.6021957
3: -30.1115532, 15.0882940, -30.1115532, 15.0882940, -45.1998444, 45.1998482
4: -28.4748173, 20.2006111, -28.4748173, 20.2006111, -48.6754303, 48.6754303
5: -24.4868717, 19.1075554, -24.4868717, 19.1075554, -43.5944290, 43.5944290
6: -22.5470924, 22.3522205, -22.5470924, 22.3522205, -44.8993111, 44.8993149
7: -24.8867416, 23.5568867, -24.8867416, 23.5568867, -48.4436264, 48.4436264
8: -34.8141861, 16.7162991, -34.8141861, 16.7162991, -51.5304832, 51.5304871
9: -21.9554176, 22.3245659, -21.9554176, 22.3245659, -44.2799835, 44.2799835

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 215

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1268250, upper bound: 43.1268240
time: 6.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1268192, upper bound: 43.1268295
time: 5.20 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -24.6925583, 19.7640648, -24.6925583, 19.7640648, -44.4566231, 44.4566231
1: -22.1655941, 17.7110901, -22.1655941, 17.7110901, -39.8766861, 39.8766861
2: -28.0165939, 17.5855999, -28.0165939, 17.5855999, -45.6021957, 45.6021957
3: -30.1115532, 15.0882940, -30.1115532, 15.0882940, -45.1998444, 45.1998482
4: -28.4748173, 20.2006111, -28.4748173, 20.2006111, -48.6754303, 48.6754303
5: -24.4868717, 19.1075554, -24.4868717, 19.1075554, -43.5944290, 43.5944290
6: -22.5470924, 22.3522205, -22.5470924, 22.3522205, -44.8993111, 44.8993149
7: -24.8867416, 23.5568867, -24.8867416, 23.5568867, -48.4436264, 48.4436264
8: -34.8141861, 16.7162991, -34.8141861, 16.7162991, -51.5304832, 51.5304871
9: -21.9554176, 22.3245659, -21.9554176, 22.3245659, -44.2799835, 44.2799835

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 95

### Candidate
type: DSZ, layer: 1, pos: 183

### Candidate
type: DSZ, layer: 1, pos: 119

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 214

### Candidate
type: DSZ, layer: 1, pos: 197

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 50

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1358346, upper bound: 43.1358319
time: 4.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1358346, upper bound: 43.1358319
time: 5.05 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -24.6925583, 19.7640648, -24.6925583, 19.7640648, -44.4566231, 44.4566231
1: -22.1655941, 17.7110901, -22.1655941, 17.7110901, -39.8766861, 39.8766861
2: -28.0165939, 17.5855999, -28.0165939, 17.5855999, -45.6021957, 45.6021957
3: -30.1115532, 15.0882940, -30.1115532, 15.0882940, -45.1998444, 45.1998482
4: -28.4748173, 20.2006111, -28.4748173, 20.2006111, -48.6754303, 48.6754303
5: -24.4868717, 19.1075554, -24.4868717, 19.1075554, -43.5944290, 43.5944290
6: -22.5470924, 22.3522205, -22.5470924, 22.3522205, -44.8993111, 44.8993149
7: -24.8867416, 23.5568867, -24.8867416, 23.5568867, -48.4436264, 48.4436264
8: -34.8141861, 16.7162991, -34.8141861, 16.7162991, -51.5304832, 51.5304871
9: -21.9554176, 22.3245659, -21.9554176, 22.3245659, -44.2799835, 44.2799835

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 117

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 250

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1268004, upper bound: 43.1268094
time: 6.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1268007, upper bound: 43.1268084
time: 5.68 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -24.6925583, 19.7640648, -24.6925583, 19.7640648, -44.4566231, 44.4566231
1: -22.1655941, 17.7110901, -22.1655941, 17.7110901, -39.8766861, 39.8766861
2: -28.0165939, 17.5855999, -28.0165939, 17.5855999, -45.6021957, 45.6021957
3: -30.1115532, 15.0882940, -30.1115532, 15.0882940, -45.1998444, 45.1998482
4: -28.4748173, 20.2006111, -28.4748173, 20.2006111, -48.6754303, 48.6754303
5: -24.4868717, 19.1075554, -24.4868717, 19.1075554, -43.5944290, 43.5944290
6: -22.5470924, 22.3522205, -22.5470924, 22.3522205, -44.8993111, 44.8993149
7: -24.8867416, 23.5568867, -24.8867416, 23.5568867, -48.4436264, 48.4436264
8: -34.8141861, 16.7162991, -34.8141861, 16.7162991, -51.5304832, 51.5304871
9: -21.9554176, 22.3245659, -21.9554176, 22.3245659, -44.2799835, 44.2799835

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1248277, upper bound: 43.1248299
time: 6.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1248277, upper bound: 43.1248299
time: 6.34 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -24.6925583, 19.7640648, -24.6925583, 19.7640648, -44.4566231, 44.4566231
1: -22.1655941, 17.7110901, -22.1655941, 17.7110901, -39.8766861, 39.8766861
2: -28.0165939, 17.5855999, -28.0165939, 17.5855999, -45.6021957, 45.6021957
3: -30.1115532, 15.0882940, -30.1115532, 15.0882940, -45.1998444, 45.1998482
4: -28.4748173, 20.2006111, -28.4748173, 20.2006111, -48.6754303, 48.6754303
5: -24.4868717, 19.1075554, -24.4868717, 19.1075554, -43.5944290, 43.5944290
6: -22.5470924, 22.3522205, -22.5470924, 22.3522205, -44.8993111, 44.8993149
7: -24.8867416, 23.5568867, -24.8867416, 23.5568867, -48.4436264, 48.4436264
8: -34.8141861, 16.7162991, -34.8141861, 16.7162991, -51.5304832, 51.5304871
9: -21.9554176, 22.3245659, -21.9554176, 22.3245659, -44.2799835, 44.2799835

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 210

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 153

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 215

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 77

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 183

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 117

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 127

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1297481, upper bound: 43.1297378
time: 5.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1297404, upper bound: 43.1297391
time: 4.87 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -24.6925583, 19.7640648, -24.6925583, 19.7640648, -44.4566231, 44.4566231
1: -22.1655941, 17.7110901, -22.1655941, 17.7110901, -39.8766861, 39.8766861
2: -28.0165939, 17.5855999, -28.0165939, 17.5855999, -45.6021957, 45.6021957
3: -30.1115532, 15.0882940, -30.1115532, 15.0882940, -45.1998444, 45.1998482
4: -28.4748173, 20.2006111, -28.4748173, 20.2006111, -48.6754303, 48.6754303
5: -24.4868717, 19.1075554, -24.4868717, 19.1075554, -43.5944290, 43.5944290
6: -22.5470924, 22.3522205, -22.5470924, 22.3522205, -44.8993111, 44.8993149
7: -24.8867416, 23.5568867, -24.8867416, 23.5568867, -48.4436264, 48.4436264
8: -34.8141861, 16.7162991, -34.8141861, 16.7162991, -51.5304832, 51.5304871
9: -21.9554176, 22.3245659, -21.9554176, 22.3245659, -44.2799835, 44.2799835

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 119

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1221947, upper bound: 43.1221933
time: 3.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1221899, upper bound: 43.1222017
time: 6.39 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 12.66 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 12.66
Output dim: 8, lower bound: -43.1297247, upper bound: 43.1297283
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 12.66
Output dim: 8, lower bound: -43.1297255, upper bound: 43.1297273
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 12.66
Output dim: 8, lower bound: -43.1297258, upper bound: 43.1297275
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 12.66
Output dim: 8, lower bound: -43.1297242, upper bound: 43.1297282
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 12.66
Output dim: 8, lower bound: -43.1318118, upper bound: 43.1318140
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 12.66
Output dim: 8, lower bound: -43.1318118, upper bound: 43.1318133
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 12.66
Output dim: 8, lower bound: -43.1317924, upper bound: 43.1317886
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 12.66
Output dim: 8, lower bound: -43.1317864, upper bound: 43.1317935
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 12.66
Output dim: 8, lower bound: -43.1360050, upper bound: 43.1360079
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 12.66
Output dim: 8, lower bound: -43.1360049, upper bound: 43.1360069
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 12.66
Output dim: 8, lower bound: -43.1341998, upper bound: 43.1342012
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 12.66
Output dim: 8, lower bound: -43.1341998, upper bound: 43.1342012
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 12.66
Output dim: 8, lower bound: -43.1268250, upper bound: 43.1268240
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 12.66
Output dim: 8, lower bound: -43.1268192, upper bound: 43.1268295
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 12.66
Output dim: 8, lower bound: -43.1358346, upper bound: 43.1358319
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 12.66
Output dim: 8, lower bound: -43.1358346, upper bound: 43.1358319
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 12.66
Output dim: 8, lower bound: -43.1268004, upper bound: 43.1268094
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 12.66
Output dim: 8, lower bound: -43.1268007, upper bound: 43.1268084
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 12.66
Output dim: 8, lower bound: -43.1248277, upper bound: 43.1248299
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 12.66
Output dim: 8, lower bound: -43.1248277, upper bound: 43.1248299
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 12.66
Output dim: 8, lower bound: -43.1297481, upper bound: 43.1297378
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 12.66
Output dim: 8, lower bound: -43.1297404, upper bound: 43.1297391
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 12.66
Output dim: 8, lower bound: -43.1221947, upper bound: 43.1221933
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 12.66
Output dim: 8, lower bound: -43.1221899, upper bound: 43.1222017

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -24.6925583, 19.7640648, -24.6925583, 19.7640648, -44.4566231, 44.4566231
1: -22.1655941, 17.7110901, -22.1655941, 17.7110901, -39.8766861, 39.8766861
2: -28.0165939, 17.5855999, -28.0165939, 17.5855999, -45.6021957, 45.6021957
3: -30.1115532, 15.0882940, -30.1115532, 15.0882940, -45.1998444, 45.1998482
4: -28.4748173, 20.2006111, -28.4748173, 20.2006111, -48.6754303, 48.6754303
5: -24.4868717, 19.1075554, -24.4868717, 19.1075554, -43.5944290, 43.5944290
6: -22.5470924, 22.3522205, -22.5470924, 22.3522205, -44.8993111, 44.8993149
7: -24.8867416, 23.5568867, -24.8867416, 23.5568867, -48.4436264, 48.4436264
8: -34.8141861, 16.7162991, -34.8141861, 16.7162991, -51.5304832, 51.5304871
9: -21.9554176, 22.3245659, -21.9554176, 22.3245659, -44.2799835, 44.2799835

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 197

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1297243, upper bound: 43.1297283
time: 5.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1297247, upper bound: 43.1297283
time: 5.70 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -24.6925583, 19.7640648, -24.6925583, 19.7640648, -44.4566231, 44.4566231
1: -22.1655941, 17.7110901, -22.1655941, 17.7110901, -39.8766861, 39.8766861
2: -28.0165939, 17.5855999, -28.0165939, 17.5855999, -45.6021957, 45.6021957
3: -30.1115532, 15.0882940, -30.1115532, 15.0882940, -45.1998444, 45.1998482
4: -28.4748173, 20.2006111, -28.4748173, 20.2006111, -48.6754303, 48.6754303
5: -24.4868717, 19.1075554, -24.4868717, 19.1075554, -43.5944290, 43.5944290
6: -22.5470924, 22.3522205, -22.5470924, 22.3522205, -44.8993111, 44.8993149
7: -24.8867416, 23.5568867, -24.8867416, 23.5568867, -48.4436264, 48.4436264
8: -34.8141861, 16.7162991, -34.8141861, 16.7162991, -51.5304832, 51.5304871
9: -21.9554176, 22.3245659, -21.9554176, 22.3245659, -44.2799835, 44.2799835

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 216

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1292366, upper bound: 43.1292341
time: 4.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1292366, upper bound: 43.1292341
time: 5.08 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -24.6925583, 19.7640648, -24.6925583, 19.7640648, -44.4566231, 44.4566231
1: -22.1655941, 17.7110901, -22.1655941, 17.7110901, -39.8766861, 39.8766861
2: -28.0165939, 17.5855999, -28.0165939, 17.5855999, -45.6021957, 45.6021957
3: -30.1115532, 15.0882940, -30.1115532, 15.0882940, -45.1998444, 45.1998482
4: -28.4748173, 20.2006111, -28.4748173, 20.2006111, -48.6754303, 48.6754303
5: -24.4868717, 19.1075554, -24.4868717, 19.1075554, -43.5944290, 43.5944290
6: -22.5470924, 22.3522205, -22.5470924, 22.3522205, -44.8993111, 44.8993149
7: -24.8867416, 23.5568867, -24.8867416, 23.5568867, -48.4436264, 48.4436264
8: -34.8141861, 16.7162991, -34.8141861, 16.7162991, -51.5304832, 51.5304871
9: -21.9554176, 22.3245659, -21.9554176, 22.3245659, -44.2799835, 44.2799835

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 153

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1148048, upper bound: 43.1148003
time: 5.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1148039, upper bound: 43.1148032
time: 5.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -24.6925583, 19.7640648, -24.6925583, 19.7640648, -44.4566231, 44.4566231
1: -22.1655941, 17.7110901, -22.1655941, 17.7110901, -39.8766861, 39.8766861
2: -28.0165939, 17.5855999, -28.0165939, 17.5855999, -45.6021957, 45.6021957
3: -30.1115532, 15.0882940, -30.1115532, 15.0882940, -45.1998444, 45.1998482
4: -28.4748173, 20.2006111, -28.4748173, 20.2006111, -48.6754303, 48.6754303
5: -24.4868717, 19.1075554, -24.4868717, 19.1075554, -43.5944290, 43.5944290
6: -22.5470924, 22.3522205, -22.5470924, 22.3522205, -44.8993111, 44.8993149
7: -24.8867416, 23.5568867, -24.8867416, 23.5568867, -48.4436264, 48.4436264
8: -34.8141861, 16.7162991, -34.8141861, 16.7162991, -51.5304832, 51.5304871
9: -21.9554176, 22.3245659, -21.9554176, 22.3245659, -44.2799835, 44.2799835

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 50

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 95

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1297240, upper bound: 43.1297282
time: 5.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1297242, upper bound: 43.1297275
time: 19.12 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 27.37 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 27.37
Output dim: 8, lower bound: -43.1297243, upper bound: 43.1297283
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 27.37
Output dim: 8, lower bound: -43.1297247, upper bound: 43.1297283
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 27.37
Output dim: 8, lower bound: -43.1292366, upper bound: 43.1292341
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 27.37
Output dim: 8, lower bound: -43.1292366, upper bound: 43.1292341
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 27.37
Output dim: 8, lower bound: -43.1148048, upper bound: 43.1148003
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 27.37
Output dim: 8, lower bound: -43.1148039, upper bound: 43.1148032
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 27.37
Output dim: 8, lower bound: -43.1297240, upper bound: 43.1297282
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 27.37
Output dim: 8, lower bound: -43.1297242, upper bound: 43.1297275
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 27.37
Output dim: 8, lower bound: -43.1318118, upper bound: 43.1318140
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 27.37
Output dim: 8, lower bound: -43.1318118, upper bound: 43.1318133
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 27.37
Output dim: 8, lower bound: -43.1317924, upper bound: 43.1317886
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 27.37
Output dim: 8, lower bound: -43.1317864, upper bound: 43.1317935
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 27.37
Output dim: 8, lower bound: -43.1360050, upper bound: 43.1360079
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 27.37
Output dim: 8, lower bound: -43.1360049, upper bound: 43.1360069
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 27.37
Output dim: 8, lower bound: -43.1341998, upper bound: 43.1342012
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 27.37
Output dim: 8, lower bound: -43.1341998, upper bound: 43.1342012
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 27.37
Output dim: 8, lower bound: -43.1358346, upper bound: 43.1358319
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 27.37
Output dim: 8, lower bound: -43.1358346, upper bound: 43.1358319
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 27.37
Output dim: 8, lower bound: -43.1297481, upper bound: 43.1297378
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 27.37
Output dim: 8, lower bound: -43.1297404, upper bound: 43.1297391

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 12.83 + 605.42 = 618.25 seconds
