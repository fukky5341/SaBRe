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
execution time: IAR + LP analysis = 1.37 + 8.10 = 9.47 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -43.1710632, upper bound: 43.1710632


# Binary Search by BASE starts (time budget: 1990.53 seconds, max iter: 100)

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
Binary search time: 36.68 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_dual_Z) starts
Time budget: 1953.85 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1405668, upper bound: 43.1405668
time: 4.61 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1405668, upper bound: 43.1405668
time: 4.61 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 9.37 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 9.37
Output dim: 8, lower bound: -43.1405668, upper bound: 43.1405668
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 9.37
Output dim: 8, lower bound: -43.1405668, upper bound: 43.1405668

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1361282, upper bound: 43.1361282
time: 5.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1361282, upper bound: 43.1361282
time: 4.50 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1361282, upper bound: 43.1361282
time: 5.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1361282, upper bound: 43.1361282
time: 4.44 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 13.38 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 13.38
Output dim: 8, lower bound: -43.1361282, upper bound: 43.1361282
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 13.38
Output dim: 8, lower bound: -43.1361282, upper bound: 43.1361282
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 13.38
Output dim: 8, lower bound: -43.1361282, upper bound: 43.1361282
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 13.38
Output dim: 8, lower bound: -43.1361282, upper bound: 43.1361282

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1361269, upper bound: 43.1361282
time: 5.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1361282, upper bound: 43.1361269
time: 8.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1361269, upper bound: 43.1361282
time: 5.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1361282, upper bound: 43.1361269
time: 5.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1361269, upper bound: 43.1361282
time: 5.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1361282, upper bound: 43.1361269
time: 8.87 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1361269, upper bound: 43.1361282
time: 5.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1361282, upper bound: 43.1361269
time: 5.29 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 16.06 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 16.06
Output dim: 8, lower bound: -43.1361269, upper bound: 43.1361282
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 16.06
Output dim: 8, lower bound: -43.1361282, upper bound: 43.1361269
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 16.06
Output dim: 8, lower bound: -43.1361269, upper bound: 43.1361282
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 16.06
Output dim: 8, lower bound: -43.1361282, upper bound: 43.1361269
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 16.06
Output dim: 8, lower bound: -43.1361269, upper bound: 43.1361282
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 16.06
Output dim: 8, lower bound: -43.1361282, upper bound: 43.1361269
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 16.06
Output dim: 8, lower bound: -43.1361269, upper bound: 43.1361282
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 16.06
Output dim: 8, lower bound: -43.1361282, upper bound: 43.1361269

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1292506, upper bound: 43.1292245
time: 5.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1292246, upper bound: 43.1292505
time: 5.10 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1292505, upper bound: 43.1292246
time: 5.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1292245, upper bound: 43.1292506
time: 5.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1292506, upper bound: 43.1292245
time: 4.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1292246, upper bound: 43.1292505
time: 5.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1292505, upper bound: 43.1292246
time: 4.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1292245, upper bound: 43.1292506
time: 4.84 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1292506, upper bound: 43.1292245
time: 5.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1292246, upper bound: 43.1292505
time: 5.15 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1292505, upper bound: 43.1292246
time: 5.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1292245, upper bound: 43.1292506
time: 5.96 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1292506, upper bound: 43.1292245
time: 4.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1292246, upper bound: 43.1292505
time: 5.45 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1292505, upper bound: 43.1292246
time: 4.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1292245, upper bound: 43.1292506
time: 4.87 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 21.76 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.76
Output dim: 8, lower bound: -43.1292506, upper bound: 43.1292245
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.76
Output dim: 8, lower bound: -43.1292246, upper bound: 43.1292505
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.76
Output dim: 8, lower bound: -43.1292505, upper bound: 43.1292246
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.76
Output dim: 8, lower bound: -43.1292245, upper bound: 43.1292506
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.76
Output dim: 8, lower bound: -43.1292506, upper bound: 43.1292245
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.76
Output dim: 8, lower bound: -43.1292246, upper bound: 43.1292505
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.76
Output dim: 8, lower bound: -43.1292505, upper bound: 43.1292246
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.76
Output dim: 8, lower bound: -43.1292245, upper bound: 43.1292506
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.76
Output dim: 8, lower bound: -43.1292506, upper bound: 43.1292245
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.76
Output dim: 8, lower bound: -43.1292246, upper bound: 43.1292505
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.76
Output dim: 8, lower bound: -43.1292505, upper bound: 43.1292246
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.76
Output dim: 8, lower bound: -43.1292245, upper bound: 43.1292506
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.76
Output dim: 8, lower bound: -43.1292506, upper bound: 43.1292245
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.76
Output dim: 8, lower bound: -43.1292246, upper bound: 43.1292505
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.76
Output dim: 8, lower bound: -43.1292505, upper bound: 43.1292246
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.76
Output dim: 8, lower bound: -43.1292245, upper bound: 43.1292506

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1189029, upper bound: 43.1188892
time: 8.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1188892, upper bound: 43.1188998
time: 6.02 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1188972, upper bound: 43.1188891
time: 5.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1188892, upper bound: 43.1189031
time: 4.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1189031, upper bound: 43.1188892
time: 8.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1188891, upper bound: 43.1188972
time: 5.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1188998, upper bound: 43.1188892
time: 4.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1188892, upper bound: 43.1189029
time: 5.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1189029, upper bound: 43.1188892
time: 5.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1188892, upper bound: 43.1188998
time: 5.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1188972, upper bound: 43.1188891
time: 3.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1188892, upper bound: 43.1189031
time: 4.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1189031, upper bound: 43.1188892
time: 5.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1188891, upper bound: 43.1188972
time: 4.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1188998, upper bound: 43.1188892
time: 4.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1188892, upper bound: 43.1189029
time: 5.18 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1189029, upper bound: 43.1188892
time: 8.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1188892, upper bound: 43.1188998
time: 6.03 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1188972, upper bound: 43.1188891
time: 5.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1188892, upper bound: 43.1189031
time: 4.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1189031, upper bound: 43.1188892
time: 9.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1188891, upper bound: 43.1188972
time: 5.21 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1188998, upper bound: 43.1188892
time: 5.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1188892, upper bound: 43.1189029
time: 5.93 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1189029, upper bound: 43.1188892
time: 5.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1188892, upper bound: 43.1188998
time: 5.66 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 22.96 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 22.96
Output dim: 8, lower bound: -43.1189029, upper bound: 43.1188892
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 22.96
Output dim: 8, lower bound: -43.1188892, upper bound: 43.1188998
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 22.96
Output dim: 8, lower bound: -43.1188972, upper bound: 43.1188891
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 22.96
Output dim: 8, lower bound: -43.1188892, upper bound: 43.1189031
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 22.96
Output dim: 8, lower bound: -43.1189031, upper bound: 43.1188892
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 22.96
Output dim: 8, lower bound: -43.1188891, upper bound: 43.1188972
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 22.96
Output dim: 8, lower bound: -43.1188998, upper bound: 43.1188892
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 22.96
Output dim: 8, lower bound: -43.1188892, upper bound: 43.1189029
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 22.96
Output dim: 8, lower bound: -43.1189029, upper bound: 43.1188892
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 22.96
Output dim: 8, lower bound: -43.1188892, upper bound: 43.1188998
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 22.96
Output dim: 8, lower bound: -43.1188972, upper bound: 43.1188891
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 22.96
Output dim: 8, lower bound: -43.1188892, upper bound: 43.1189031
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 22.96
Output dim: 8, lower bound: -43.1189031, upper bound: 43.1188892
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 22.96
Output dim: 8, lower bound: -43.1188891, upper bound: 43.1188972
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 22.96
Output dim: 8, lower bound: -43.1188998, upper bound: 43.1188892
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 22.96
Output dim: 8, lower bound: -43.1188892, upper bound: 43.1189029
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 22.96
Output dim: 8, lower bound: -43.1189029, upper bound: 43.1188892
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 22.96
Output dim: 8, lower bound: -43.1188892, upper bound: 43.1188998
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 22.96
Output dim: 8, lower bound: -43.1188972, upper bound: 43.1188891
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 22.96
Output dim: 8, lower bound: -43.1188892, upper bound: 43.1189031
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 22.96
Output dim: 8, lower bound: -43.1189031, upper bound: 43.1188892
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 22.96
Output dim: 8, lower bound: -43.1188891, upper bound: 43.1188972
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 22.96
Output dim: 8, lower bound: -43.1188998, upper bound: 43.1188892
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 22.96
Output dim: 8, lower bound: -43.1188892, upper bound: 43.1189029
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 22.96
Output dim: 8, lower bound: -43.1189029, upper bound: 43.1188892
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 22.96
Output dim: 8, lower bound: -43.1188892, upper bound: 43.1188998
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.96
Output dim: 8, lower bound: -43.1292246, upper bound: 43.1292505
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.96
Output dim: 8, lower bound: -43.1292505, upper bound: 43.1292246
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.96
Output dim: 8, lower bound: -43.1292245, upper bound: 43.1292506
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=51.530487060546875
rel_dist={8: [-43.17090795605019, 43.17090794479026]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1405371, upper bound: 43.1405371
time: 5.86 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1405371, upper bound: 43.1405371
time: 5.89 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 11.91 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 11.91
Output dim: 8, lower bound: -43.1405371, upper bound: 43.1405371
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 11.91
Output dim: 8, lower bound: -43.1405371, upper bound: 43.1405371

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1361130, upper bound: 43.1361130
time: 5.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1361130, upper bound: 43.1361130
time: 5.29 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1361130, upper bound: 43.1361130
time: 5.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1361130, upper bound: 43.1361130
time: 5.29 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 14.06 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 14.06
Output dim: 8, lower bound: -43.1361130, upper bound: 43.1361130
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 14.06
Output dim: 8, lower bound: -43.1361130, upper bound: 43.1361130
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 14.06
Output dim: 8, lower bound: -43.1361130, upper bound: 43.1361130
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 14.06
Output dim: 8, lower bound: -43.1361130, upper bound: 43.1361130

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1361123, upper bound: 43.1361130
time: 5.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1361130, upper bound: 43.1361123
time: 5.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1361123, upper bound: 43.1361130
time: 4.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1361130, upper bound: 43.1361123
time: 6.10 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1361123, upper bound: 43.1361130
time: 5.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1361130, upper bound: 43.1361123
time: 5.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1361123, upper bound: 43.1361130
time: 4.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1361130, upper bound: 43.1361123
time: 6.09 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 16.70 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 16.70
Output dim: 8, lower bound: -43.1361123, upper bound: 43.1361130
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 16.70
Output dim: 8, lower bound: -43.1361130, upper bound: 43.1361123
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 16.70
Output dim: 8, lower bound: -43.1361123, upper bound: 43.1361130
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 16.70
Output dim: 8, lower bound: -43.1361130, upper bound: 43.1361123
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 16.70
Output dim: 8, lower bound: -43.1361123, upper bound: 43.1361130
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 16.70
Output dim: 8, lower bound: -43.1361130, upper bound: 43.1361123
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 16.70
Output dim: 8, lower bound: -43.1361123, upper bound: 43.1361130
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 16.70
Output dim: 8, lower bound: -43.1361130, upper bound: 43.1361123

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1292215, upper bound: 43.1292085
time: 5.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1292085, upper bound: 43.1292215
time: 5.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1292215, upper bound: 43.1292085
time: 6.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1292085, upper bound: 43.1292215
time: 5.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1292215, upper bound: 43.1292085
time: 5.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1292085, upper bound: 43.1292215
time: 6.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1292215, upper bound: 43.1292085
time: 6.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1292085, upper bound: 43.1292215
time: 5.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1292215, upper bound: 43.1292085
time: 5.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1292085, upper bound: 43.1292215
time: 7.46 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1292215, upper bound: 43.1292085
time: 6.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1292085, upper bound: 43.1292215
time: 5.94 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1292215, upper bound: 43.1292085
time: 6.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1292085, upper bound: 43.1292215
time: 5.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1292215, upper bound: 43.1292085
time: 6.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1292085, upper bound: 43.1292215
time: 5.73 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 24.11 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.11
Output dim: 8, lower bound: -43.1292215, upper bound: 43.1292085
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.11
Output dim: 8, lower bound: -43.1292085, upper bound: 43.1292215
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.11
Output dim: 8, lower bound: -43.1292215, upper bound: 43.1292085
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.11
Output dim: 8, lower bound: -43.1292085, upper bound: 43.1292215
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.11
Output dim: 8, lower bound: -43.1292215, upper bound: 43.1292085
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.11
Output dim: 8, lower bound: -43.1292085, upper bound: 43.1292215
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.11
Output dim: 8, lower bound: -43.1292215, upper bound: 43.1292085
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.11
Output dim: 8, lower bound: -43.1292085, upper bound: 43.1292215
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.11
Output dim: 8, lower bound: -43.1292215, upper bound: 43.1292085
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.11
Output dim: 8, lower bound: -43.1292085, upper bound: 43.1292215
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.11
Output dim: 8, lower bound: -43.1292215, upper bound: 43.1292085
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.11
Output dim: 8, lower bound: -43.1292085, upper bound: 43.1292215
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.11
Output dim: 8, lower bound: -43.1292215, upper bound: 43.1292085
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.11
Output dim: 8, lower bound: -43.1292085, upper bound: 43.1292215
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.11
Output dim: 8, lower bound: -43.1292215, upper bound: 43.1292085
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.11
Output dim: 8, lower bound: -43.1292085, upper bound: 43.1292215

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1188661, upper bound: 43.1188547
time: 6.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1188548, upper bound: 43.1188635
time: 4.51 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1188625, upper bound: 43.1188547
time: 5.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1188548, upper bound: 43.1188667
time: 5.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1188667, upper bound: 43.1188548
time: 5.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1188547, upper bound: 43.1188625
time: 4.21 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1188635, upper bound: 43.1188548
time: 5.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1188547, upper bound: 43.1188661
time: 4.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1188661, upper bound: 43.1188547
time: 6.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1188548, upper bound: 43.1188635
time: 11.13 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1188625, upper bound: 43.1188547
time: 3.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1188548, upper bound: 43.1188667
time: 4.42 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1188667, upper bound: 43.1188548
time: 5.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1188547, upper bound: 43.1188625
time: 6.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1188635, upper bound: 43.1188548
time: 6.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1188547, upper bound: 43.1188661
time: 5.45 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1188661, upper bound: 43.1188547
time: 6.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1188548, upper bound: 43.1188635
time: 4.47 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1188625, upper bound: 43.1188547
time: 5.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1188548, upper bound: 43.1188667
time: 5.49 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1188667, upper bound: 43.1188548
time: 5.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1188547, upper bound: 43.1188625
time: 4.26 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1188635, upper bound: 43.1188548
time: 6.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1188547, upper bound: 43.1188661
time: 4.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=51.530487060546875
rel_dist={8: [-43.17074367834749, 43.17074366907076]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1404961, upper bound: 43.1404961
time: 6.15 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1404961, upper bound: 43.1404961
time: 5.92 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 12.23 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 12.23
Output dim: 8, lower bound: -43.1404961, upper bound: 43.1404961
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 12.23
Output dim: 8, lower bound: -43.1404961, upper bound: 43.1404961

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1360787, upper bound: 43.1360787
time: 6.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1360787, upper bound: 43.1360787
time: 6.75 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1360787, upper bound: 43.1360787
time: 6.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1360787, upper bound: 43.1360787
time: 7.05 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 19.16 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 19.16
Output dim: 8, lower bound: -43.1360787, upper bound: 43.1360787
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 19.16
Output dim: 8, lower bound: -43.1360787, upper bound: 43.1360787
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 19.16
Output dim: 8, lower bound: -43.1360787, upper bound: 43.1360787
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 19.16
Output dim: 8, lower bound: -43.1360787, upper bound: 43.1360787

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1360787, upper bound: 43.1360787
time: 5.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1360787, upper bound: 43.1360787
time: 7.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1360787, upper bound: 43.1360787
time: 14.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1360787, upper bound: 43.1360787
time: 7.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1360787, upper bound: 43.1360787
time: 5.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1360787, upper bound: 43.1360787
time: 7.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1360787, upper bound: 43.1360787
time: 16.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1360787, upper bound: 43.1360787
time: 7.34 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 30.81 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 30.81
Output dim: 8, lower bound: -43.1360787, upper bound: 43.1360787
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 30.81
Output dim: 8, lower bound: -43.1360787, upper bound: 43.1360787
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 30.81
Output dim: 8, lower bound: -43.1360787, upper bound: 43.1360787
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 30.81
Output dim: 8, lower bound: -43.1360787, upper bound: 43.1360787
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 30.81
Output dim: 8, lower bound: -43.1360787, upper bound: 43.1360787
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 30.81
Output dim: 8, lower bound: -43.1360787, upper bound: 43.1360787
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 30.81
Output dim: 8, lower bound: -43.1360787, upper bound: 43.1360787
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 30.81
Output dim: 8, lower bound: -43.1360787, upper bound: 43.1360787

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1291941, upper bound: 43.1291898
time: 14.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1291898, upper bound: 43.1291941
time: 8.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1291941, upper bound: 43.1291898
time: 5.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1291898, upper bound: 43.1291941
time: 7.15 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1291941, upper bound: 43.1291898
time: 11.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1291898, upper bound: 43.1291941
time: 15.12 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1291941, upper bound: 43.1291898
time: 5.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1291898, upper bound: 43.1291941
time: 4.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1291941, upper bound: 43.1291898
time: 16.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1291898, upper bound: 43.1291941
time: 8.90 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1291941, upper bound: 43.1291898
time: 5.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1291898, upper bound: 43.1291941
time: 7.08 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1291941, upper bound: 43.1291898
time: 11.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1291898, upper bound: 43.1291941
time: 16.20 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1291941, upper bound: 43.1291898
time: 5.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1291898, upper bound: 43.1291941
time: 4.60 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 22.57 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.57
Output dim: 8, lower bound: -43.1291941, upper bound: 43.1291898
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.57
Output dim: 8, lower bound: -43.1291898, upper bound: 43.1291941
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.57
Output dim: 8, lower bound: -43.1291941, upper bound: 43.1291898
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.57
Output dim: 8, lower bound: -43.1291898, upper bound: 43.1291941
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.57
Output dim: 8, lower bound: -43.1291941, upper bound: 43.1291898
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.57
Output dim: 8, lower bound: -43.1291898, upper bound: 43.1291941
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.57
Output dim: 8, lower bound: -43.1291941, upper bound: 43.1291898
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.57
Output dim: 8, lower bound: -43.1291898, upper bound: 43.1291941
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.57
Output dim: 8, lower bound: -43.1291941, upper bound: 43.1291898
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.57
Output dim: 8, lower bound: -43.1291898, upper bound: 43.1291941
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.57
Output dim: 8, lower bound: -43.1291941, upper bound: 43.1291898
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.57
Output dim: 8, lower bound: -43.1291898, upper bound: 43.1291941
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.57
Output dim: 8, lower bound: -43.1291941, upper bound: 43.1291898
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.57
Output dim: 8, lower bound: -43.1291898, upper bound: 43.1291941
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.57
Output dim: 8, lower bound: -43.1291941, upper bound: 43.1291898
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.57
Output dim: 8, lower bound: -43.1291898, upper bound: 43.1291941

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1188024, upper bound: 43.1187983
time: 7.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1187985, upper bound: 43.1188016
time: 5.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1188013, upper bound: 43.1187986
time: 11.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1187983, upper bound: 43.1188026
time: 5.26 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1188026, upper bound: 43.1187983
time: 5.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1187986, upper bound: 43.1188013
time: 6.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1188016, upper bound: 43.1187985
time: 16.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1187983, upper bound: 43.1188024
time: 6.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1188024, upper bound: 43.1187983
time: 6.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1187985, upper bound: 43.1188016
time: 4.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1188013, upper bound: 43.1187986
time: 10.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1187983, upper bound: 43.1188026
time: 10.23 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1188026, upper bound: 43.1187983
time: 6.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1187986, upper bound: 43.1188013
time: 6.32 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 24.60 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.60
Output dim: 8, lower bound: -43.1188024, upper bound: 43.1187983
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.60
Output dim: 8, lower bound: -43.1187985, upper bound: 43.1188016
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.60
Output dim: 8, lower bound: -43.1188013, upper bound: 43.1187986
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.60
Output dim: 8, lower bound: -43.1187983, upper bound: 43.1188026
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.60
Output dim: 8, lower bound: -43.1188026, upper bound: 43.1187983
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.60
Output dim: 8, lower bound: -43.1187986, upper bound: 43.1188013
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.60
Output dim: 8, lower bound: -43.1188016, upper bound: 43.1187985
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.60
Output dim: 8, lower bound: -43.1187983, upper bound: 43.1188024
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.60
Output dim: 8, lower bound: -43.1188024, upper bound: 43.1187983
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.60
Output dim: 8, lower bound: -43.1187985, upper bound: 43.1188016
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.60
Output dim: 8, lower bound: -43.1188013, upper bound: 43.1187986
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.60
Output dim: 8, lower bound: -43.1187983, upper bound: 43.1188026
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.60
Output dim: 8, lower bound: -43.1188026, upper bound: 43.1187983
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.60
Output dim: 8, lower bound: -43.1187986, upper bound: 43.1188013
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.60
Output dim: 8, lower bound: -43.1291898, upper bound: 43.1291941
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.60
Output dim: 8, lower bound: -43.1291941, upper bound: 43.1291898
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.60
Output dim: 8, lower bound: -43.1291898, upper bound: 43.1291941
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.60
Output dim: 8, lower bound: -43.1291941, upper bound: 43.1291898
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.60
Output dim: 8, lower bound: -43.1291898, upper bound: 43.1291941
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.60
Output dim: 8, lower bound: -43.1291941, upper bound: 43.1291898
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.60
Output dim: 8, lower bound: -43.1291898, upper bound: 43.1291941
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.60
Output dim: 8, lower bound: -43.1291941, upper bound: 43.1291898
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.60
Output dim: 8, lower bound: -43.1291898, upper bound: 43.1291941
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=51.530487060546875
rel_dist={8: [-43.1705810903977, 43.17058109163787]}

## Binary Search with RS_dual_Z Result
status: None
Maximum delta epsilon: None
execution time: 1811.42 seconds
