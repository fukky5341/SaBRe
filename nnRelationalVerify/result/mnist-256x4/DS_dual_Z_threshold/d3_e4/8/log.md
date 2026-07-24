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
execution time: IAR + RelationalAnalysis = 0.78 + 11.96 = 12.75 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -43.1707437, upper bound: 43.1707437

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1405371, upper bound: 43.1405371
time: 5.56 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1405371, upper bound: 43.1405371
time: 5.51 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 11.14 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 11.14
Output dim: 8, lower bound: -43.1405371, upper bound: 43.1405371
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 11.14
Output dim: 8, lower bound: -43.1405371, upper bound: 43.1405371

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1361130, upper bound: 43.1361130
time: 4.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1361130, upper bound: 43.1361130
time: 5.14 seconds

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

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1361130, upper bound: 43.1361130
time: 5.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1361130, upper bound: 43.1361130
time: 5.16 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 13.09 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 13.09
Output dim: 8, lower bound: -43.1361130, upper bound: 43.1361130
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 13.09
Output dim: 8, lower bound: -43.1361130, upper bound: 43.1361130
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 13.09
Output dim: 8, lower bound: -43.1361130, upper bound: 43.1361130
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 13.09
Output dim: 8, lower bound: -43.1361130, upper bound: 43.1361130

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 184

### Candidate
type: DSZ, layer: 1, pos: 215

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1361123, upper bound: 43.1361130
time: 4.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1361130, upper bound: 43.1361123
time: 5.05 seconds

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 184

### Candidate
type: DSZ, layer: 1, pos: 215

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1361123, upper bound: 43.1361130
time: 4.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1361130, upper bound: 43.1361123
time: 5.74 seconds

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 184

### Candidate
type: DSZ, layer: 1, pos: 215

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1361123, upper bound: 43.1361130
time: 4.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1361130, upper bound: 43.1361123
time: 4.85 seconds

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 184

### Candidate
type: DSZ, layer: 1, pos: 215

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1361123, upper bound: 43.1361130
time: 4.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1361130, upper bound: 43.1361123
time: 5.73 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 12.89 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 12.89
Output dim: 8, lower bound: -43.1361123, upper bound: 43.1361130
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 12.89
Output dim: 8, lower bound: -43.1361130, upper bound: 43.1361123
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 12.89
Output dim: 8, lower bound: -43.1361123, upper bound: 43.1361130
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 12.89
Output dim: 8, lower bound: -43.1361130, upper bound: 43.1361123
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 12.89
Output dim: 8, lower bound: -43.1361123, upper bound: 43.1361130
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 12.89
Output dim: 8, lower bound: -43.1361130, upper bound: 43.1361123
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 12.89
Output dim: 8, lower bound: -43.1361123, upper bound: 43.1361130
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 12.89
Output dim: 8, lower bound: -43.1361130, upper bound: 43.1361123

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 184

### Candidate
type: DSZ, layer: 1, pos: 215

### Candidate
type: DSZ, layer: 1, pos: 219

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 153

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1292215, upper bound: 43.1292085
time: 5.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1292085, upper bound: 43.1292215
time: 5.50 seconds

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 184

### Candidate
type: DSZ, layer: 1, pos: 215

### Candidate
type: DSZ, layer: 1, pos: 219

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 153

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1292215, upper bound: 43.1292085
time: 6.00 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1292085, upper bound: 43.1292215
time: 5.61 seconds

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 184

### Candidate
type: DSZ, layer: 1, pos: 215

### Candidate
type: DSZ, layer: 1, pos: 219

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 153

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1292215, upper bound: 43.1292085
time: 5.09 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1292085, upper bound: 43.1292215
time: 6.37 seconds

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 184

### Candidate
type: DSZ, layer: 1, pos: 215

### Candidate
type: DSZ, layer: 1, pos: 219

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 153

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1292215, upper bound: 43.1292085
time: 6.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1292085, upper bound: 43.1292215
time: 5.41 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 184

### Candidate
type: DSZ, layer: 1, pos: 215

### Candidate
type: DSZ, layer: 1, pos: 219

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 153

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1292215, upper bound: 43.1292085
time: 5.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1292085, upper bound: 43.1292215
time: 6.92 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 184

### Candidate
type: DSZ, layer: 1, pos: 215

### Candidate
type: DSZ, layer: 1, pos: 219

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 153

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1292215, upper bound: 43.1292085
time: 6.34 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1292085, upper bound: 43.1292215
time: 5.71 seconds

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 184

### Candidate
type: DSZ, layer: 1, pos: 215

### Candidate
type: DSZ, layer: 1, pos: 219

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 153

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1292215, upper bound: 43.1292085
time: 5.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1292085, upper bound: 43.1292215
time: 5.41 seconds

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 184

### Candidate
type: DSZ, layer: 1, pos: 215

### Candidate
type: DSZ, layer: 1, pos: 219

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 153

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1292215, upper bound: 43.1292085
time: 5.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1292085, upper bound: 43.1292215
time: 5.37 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 17.74 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 17.74
Output dim: 8, lower bound: -43.1292215, upper bound: 43.1292085
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 17.74
Output dim: 8, lower bound: -43.1292085, upper bound: 43.1292215
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 17.74
Output dim: 8, lower bound: -43.1292215, upper bound: 43.1292085
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 17.74
Output dim: 8, lower bound: -43.1292085, upper bound: 43.1292215
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 17.74
Output dim: 8, lower bound: -43.1292215, upper bound: 43.1292085
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 17.74
Output dim: 8, lower bound: -43.1292085, upper bound: 43.1292215
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 17.74
Output dim: 8, lower bound: -43.1292215, upper bound: 43.1292085
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 17.74
Output dim: 8, lower bound: -43.1292085, upper bound: 43.1292215
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 17.74
Output dim: 8, lower bound: -43.1292215, upper bound: 43.1292085
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 17.74
Output dim: 8, lower bound: -43.1292085, upper bound: 43.1292215
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 17.74
Output dim: 8, lower bound: -43.1292215, upper bound: 43.1292085
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 17.74
Output dim: 8, lower bound: -43.1292085, upper bound: 43.1292215
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 17.74
Output dim: 8, lower bound: -43.1292215, upper bound: 43.1292085
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 17.74
Output dim: 8, lower bound: -43.1292085, upper bound: 43.1292215
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 17.74
Output dim: 8, lower bound: -43.1292215, upper bound: 43.1292085
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 17.74
Output dim: 8, lower bound: -43.1292085, upper bound: 43.1292215

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 184

### Candidate
type: DSZ, layer: 1, pos: 215

### Candidate
type: DSZ, layer: 1, pos: 219

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 153

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1188661, upper bound: 43.1188547
time: 6.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1188548, upper bound: 43.1188635
time: 4.32 seconds

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 184

### Candidate
type: DSZ, layer: 1, pos: 215

### Candidate
type: DSZ, layer: 1, pos: 219

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 153

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1188625, upper bound: 43.1188547
time: 4.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1188548, upper bound: 43.1188667
time: 5.26 seconds

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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 184

### Candidate
type: DSZ, layer: 1, pos: 215

### Candidate
type: DSZ, layer: 1, pos: 219

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 153

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1188667, upper bound: 43.1188548
time: 5.13 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1188547, upper bound: 43.1188625
time: 4.01 seconds

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 184

### Candidate
type: DSZ, layer: 1, pos: 215

### Candidate
type: DSZ, layer: 1, pos: 219

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 153

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1188635, upper bound: 43.1188548
time: 5.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1188547, upper bound: 43.1188661
time: 4.48 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

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
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 184

### Candidate
type: DSZ, layer: 1, pos: 215

### Candidate
type: DSZ, layer: 1, pos: 219

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 153

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1188661, upper bound: 43.1188547
time: 6.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1188548, upper bound: 43.1188635
time: 11.08 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

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
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 184

### Candidate
type: DSZ, layer: 1, pos: 215

### Candidate
type: DSZ, layer: 1, pos: 219

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 153

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1188625, upper bound: 43.1188547
time: 3.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1188548, upper bound: 43.1188667
time: 4.15 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

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
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 184

### Candidate
type: DSZ, layer: 1, pos: 215

### Candidate
type: DSZ, layer: 1, pos: 219

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 153

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1188667, upper bound: 43.1188548
time: 4.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1188547, upper bound: 43.1188625
time: 5.92 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 184

### Candidate
type: DSZ, layer: 1, pos: 215

### Candidate
type: DSZ, layer: 1, pos: 219

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 153

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1188635, upper bound: 43.1188548
time: 6.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1188547, upper bound: 43.1188661
time: 5.02 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 184

### Candidate
type: DSZ, layer: 1, pos: 215

### Candidate
type: DSZ, layer: 1, pos: 219

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 153

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1188661, upper bound: 43.1188547
time: 6.12 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1188548, upper bound: 43.1188635
time: 4.20 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 184

### Candidate
type: DSZ, layer: 1, pos: 215

### Candidate
type: DSZ, layer: 1, pos: 219

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 153

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1188625, upper bound: 43.1188547
time: 4.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1188548, upper bound: 43.1188667
time: 5.11 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

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
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 184

### Candidate
type: DSZ, layer: 1, pos: 215

### Candidate
type: DSZ, layer: 1, pos: 219

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 153

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1188667, upper bound: 43.1188548
time: 4.92 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1188547, upper bound: 43.1188625
time: 3.96 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 184

### Candidate
type: DSZ, layer: 1, pos: 215

### Candidate
type: DSZ, layer: 1, pos: 219

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 153

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1188635, upper bound: 43.1188548
time: 5.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1188547, upper bound: 43.1188661
time: 4.37 seconds

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 184

### Candidate
type: DSZ, layer: 1, pos: 215

### Candidate
type: DSZ, layer: 1, pos: 219

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 153

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1188661, upper bound: 43.1188547
time: 6.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1188548, upper bound: 43.1188635
time: 11.04 seconds

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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 184

### Candidate
type: DSZ, layer: 1, pos: 215

### Candidate
type: DSZ, layer: 1, pos: 219

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 153

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1188625, upper bound: 43.1188547
time: 3.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1188548, upper bound: 43.1188667
time: 4.13 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

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
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 184

### Candidate
type: DSZ, layer: 1, pos: 215

### Candidate
type: DSZ, layer: 1, pos: 219

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 153

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1188667, upper bound: 43.1188548
time: 4.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1188547, upper bound: 43.1188625
time: 6.61 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

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
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 184

### Candidate
type: DSZ, layer: 1, pos: 215

### Candidate
type: DSZ, layer: 1, pos: 219

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 153

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1188635, upper bound: 43.1188548
time: 6.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1188547, upper bound: 43.1188661
time: 5.13 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 12.48 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 12.48
Output dim: 8, lower bound: -43.1188661, upper bound: 43.1188547
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 12.48
Output dim: 8, lower bound: -43.1188548, upper bound: 43.1188635
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 12.48
Output dim: 8, lower bound: -43.1188625, upper bound: 43.1188547
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 12.48
Output dim: 8, lower bound: -43.1188548, upper bound: 43.1188667
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 12.48
Output dim: 8, lower bound: -43.1188667, upper bound: 43.1188548
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 12.48
Output dim: 8, lower bound: -43.1188547, upper bound: 43.1188625
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 12.48
Output dim: 8, lower bound: -43.1188635, upper bound: 43.1188548
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 12.48
Output dim: 8, lower bound: -43.1188547, upper bound: 43.1188661
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 12.48
Output dim: 8, lower bound: -43.1188661, upper bound: 43.1188547
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 12.48
Output dim: 8, lower bound: -43.1188548, upper bound: 43.1188635
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 12.48
Output dim: 8, lower bound: -43.1188625, upper bound: 43.1188547
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 12.48
Output dim: 8, lower bound: -43.1188548, upper bound: 43.1188667
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 12.48
Output dim: 8, lower bound: -43.1188667, upper bound: 43.1188548
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 12.48
Output dim: 8, lower bound: -43.1188547, upper bound: 43.1188625
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 12.48
Output dim: 8, lower bound: -43.1188635, upper bound: 43.1188548
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 12.48
Output dim: 8, lower bound: -43.1188547, upper bound: 43.1188661
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 12.48
Output dim: 8, lower bound: -43.1188661, upper bound: 43.1188547
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 12.48
Output dim: 8, lower bound: -43.1188548, upper bound: 43.1188635
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 12.48
Output dim: 8, lower bound: -43.1188625, upper bound: 43.1188547
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 12.48
Output dim: 8, lower bound: -43.1188548, upper bound: 43.1188667
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 12.48
Output dim: 8, lower bound: -43.1188667, upper bound: 43.1188548
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 12.48
Output dim: 8, lower bound: -43.1188547, upper bound: 43.1188625
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 12.48
Output dim: 8, lower bound: -43.1188635, upper bound: 43.1188548
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 12.48
Output dim: 8, lower bound: -43.1188547, upper bound: 43.1188661
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 12.48
Output dim: 8, lower bound: -43.1188661, upper bound: 43.1188547
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 12.48
Output dim: 8, lower bound: -43.1188548, upper bound: 43.1188635
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 12.48
Output dim: 8, lower bound: -43.1188625, upper bound: 43.1188547
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 12.48
Output dim: 8, lower bound: -43.1188548, upper bound: 43.1188667
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 12.48
Output dim: 8, lower bound: -43.1188667, upper bound: 43.1188548
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 12.48
Output dim: 8, lower bound: -43.1188547, upper bound: 43.1188625
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 12.48
Output dim: 8, lower bound: -43.1188635, upper bound: 43.1188548
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 12.48
Output dim: 8, lower bound: -43.1188547, upper bound: 43.1188661

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 12.75 + 419.70 = 432.45 seconds
