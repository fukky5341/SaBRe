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
execution time: IAR + LP analysis = 1.32 + 8.01 = 9.33 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -43.1710632, upper bound: 43.1710632


# Binary Search by BASE starts (time budget: 1990.67 seconds, max iter: 100)

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
Binary search time: 36.11 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_random_Z) starts
Time budget: 1954.55 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 175

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1663026, upper bound: 43.1663026
time: 6.14 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1663026, upper bound: 43.1663026
time: 6.10 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 12.26 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 12.26
Output dim: 8, lower bound: -43.1663026, upper bound: 43.1663026
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 12.26
Output dim: 8, lower bound: -43.1663026, upper bound: 43.1663026

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

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1524085, upper bound: 43.1524085
time: 6.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1524085, upper bound: 43.1524085
time: 6.38 seconds

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

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1579272, upper bound: 43.1579272
time: 6.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1579272, upper bound: 43.1579272
time: 5.30 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 13.19 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 13.19
Output dim: 8, lower bound: -43.1524085, upper bound: 43.1524085
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 13.19
Output dim: 8, lower bound: -43.1524085, upper bound: 43.1524085
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 13.19
Output dim: 8, lower bound: -43.1579272, upper bound: 43.1579272
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 13.19
Output dim: 8, lower bound: -43.1579272, upper bound: 43.1579272

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

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1459375, upper bound: 43.1459277
time: 6.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1459277, upper bound: 43.1459375
time: 5.48 seconds

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

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 119

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1366936, upper bound: 43.1366936
time: 6.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1366936, upper bound: 43.1366936
time: 5.94 seconds

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

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1579272, upper bound: 43.1579270
time: 7.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1579270, upper bound: 43.1579272
time: 6.77 seconds

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

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1579272, upper bound: 43.1579163
time: 6.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1579163, upper bound: 43.1579272
time: 5.46 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 15.19 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 15.19
Output dim: 8, lower bound: -43.1459375, upper bound: 43.1459277
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 15.19
Output dim: 8, lower bound: -43.1459277, upper bound: 43.1459375
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 15.19
Output dim: 8, lower bound: -43.1366936, upper bound: 43.1366936
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 15.19
Output dim: 8, lower bound: -43.1366936, upper bound: 43.1366936
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 15.19
Output dim: 8, lower bound: -43.1579272, upper bound: 43.1579270
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 15.19
Output dim: 8, lower bound: -43.1579270, upper bound: 43.1579272
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 15.19
Output dim: 8, lower bound: -43.1579272, upper bound: 43.1579163
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 15.19
Output dim: 8, lower bound: -43.1579163, upper bound: 43.1579272

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 119

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1360941, upper bound: 43.1360621
time: 4.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1360941, upper bound: 43.1360621
time: 21.04 seconds

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

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1378492, upper bound: 43.1378087
time: 5.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1377933, upper bound: 43.1378521
time: 6.30 seconds

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

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1293805, upper bound: 43.1293657
time: 5.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1293657, upper bound: 43.1293805
time: 5.83 seconds

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

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1366936, upper bound: 43.1366929
time: 5.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1366929, upper bound: 43.1366936
time: 5.82 seconds

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

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1575019, upper bound: 43.1575018
time: 5.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1575019, upper bound: 43.1575018
time: 6.92 seconds

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1265648, upper bound: 43.1265645
time: 11.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1265648, upper bound: 43.1265645
time: 8.22 seconds

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1487691, upper bound: 43.1487619
time: 6.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1487691, upper bound: 43.1487619
time: 6.83 seconds

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
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 119

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1508705, upper bound: 43.1508721
time: 7.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1508705, upper bound: 43.1508721
time: 6.76 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 17.49 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.49
Output dim: 8, lower bound: -43.1360941, upper bound: 43.1360621
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.49
Output dim: 8, lower bound: -43.1360941, upper bound: 43.1360621
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.49
Output dim: 8, lower bound: -43.1378492, upper bound: 43.1378087
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.49
Output dim: 8, lower bound: -43.1377933, upper bound: 43.1378521
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.49
Output dim: 8, lower bound: -43.1293805, upper bound: 43.1293657
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.49
Output dim: 8, lower bound: -43.1293657, upper bound: 43.1293805
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.49
Output dim: 8, lower bound: -43.1366936, upper bound: 43.1366929
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.49
Output dim: 8, lower bound: -43.1366929, upper bound: 43.1366936
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.49
Output dim: 8, lower bound: -43.1575019, upper bound: 43.1575018
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.49
Output dim: 8, lower bound: -43.1575019, upper bound: 43.1575018
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 17.49
Output dim: 8, lower bound: -43.1265648, upper bound: 43.1265645
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 17.49
Output dim: 8, lower bound: -43.1265648, upper bound: 43.1265645
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.49
Output dim: 8, lower bound: -43.1487691, upper bound: 43.1487619
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.49
Output dim: 8, lower bound: -43.1487691, upper bound: 43.1487619
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.49
Output dim: 8, lower bound: -43.1508705, upper bound: 43.1508721
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.49
Output dim: 8, lower bound: -43.1508705, upper bound: 43.1508721

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

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1284893, upper bound: 43.1284993
time: 6.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1285070, upper bound: 43.1284817
time: 6.52 seconds

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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1185777, upper bound: 43.1185777
time: 9.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1185777, upper bound: 43.1185777
time: 10.20 seconds

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

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1348179, upper bound: 43.1347880
time: 6.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1348179, upper bound: 43.1347880
time: 6.93 seconds

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

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1318268, upper bound: 43.1318979
time: 6.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1318268, upper bound: 43.1318979
time: 6.93 seconds

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

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1293805, upper bound: 43.1293369
time: 5.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1293501, upper bound: 43.1293657
time: 4.61 seconds

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

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 117

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1293626, upper bound: 43.1293805
time: 4.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1293657, upper bound: 43.1293794
time: 10.01 seconds

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
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1358385, upper bound: 43.1358550
time: 4.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1358563, upper bound: 43.1358359
time: 5.06 seconds

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

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1366917, upper bound: 43.1366936
time: 5.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1366929, upper bound: 43.1366918
time: 14.95 seconds

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

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1557949, upper bound: 43.1557917
time: 6.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1557949, upper bound: 43.1557917
time: 6.43 seconds

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

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1455482, upper bound: 43.1455479
time: 6.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1455482, upper bound: 43.1455479
time: 6.77 seconds

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

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1406803, upper bound: 43.1406685
time: 6.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1406803, upper bound: 43.1406685
time: 6.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 117

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1297703, upper bound: 43.1297648
time: 6.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1297703, upper bound: 43.1297648
time: 6.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1337004, upper bound: 43.1337115
time: 9.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1337004, upper bound: 43.1337115
time: 11.28 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

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

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1324350, upper bound: 43.1324354
time: 17.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1324350, upper bound: 43.1324354
time: 13.51 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 32.82 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 32.82
Output dim: 8, lower bound: -43.1284893, upper bound: 43.1284993
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 32.82
Output dim: 8, lower bound: -43.1285070, upper bound: 43.1284817
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 32.82
Output dim: 8, lower bound: -43.1185777, upper bound: 43.1185777
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 32.82
Output dim: 8, lower bound: -43.1185777, upper bound: 43.1185777
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 32.82
Output dim: 8, lower bound: -43.1348179, upper bound: 43.1347880
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 32.82
Output dim: 8, lower bound: -43.1348179, upper bound: 43.1347880
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 32.82
Output dim: 8, lower bound: -43.1318268, upper bound: 43.1318979
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 32.82
Output dim: 8, lower bound: -43.1318268, upper bound: 43.1318979
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 32.82
Output dim: 8, lower bound: -43.1293805, upper bound: 43.1293369
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 32.82
Output dim: 8, lower bound: -43.1293501, upper bound: 43.1293657
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 32.82
Output dim: 8, lower bound: -43.1293626, upper bound: 43.1293805
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 32.82
Output dim: 8, lower bound: -43.1293657, upper bound: 43.1293794
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 32.82
Output dim: 8, lower bound: -43.1358385, upper bound: 43.1358550
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 32.82
Output dim: 8, lower bound: -43.1358563, upper bound: 43.1358359
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 32.82
Output dim: 8, lower bound: -43.1366917, upper bound: 43.1366936
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 32.82
Output dim: 8, lower bound: -43.1366929, upper bound: 43.1366918
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 32.82
Output dim: 8, lower bound: -43.1557949, upper bound: 43.1557917
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 32.82
Output dim: 8, lower bound: -43.1557949, upper bound: 43.1557917
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 32.82
Output dim: 8, lower bound: -43.1455482, upper bound: 43.1455479
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 32.82
Output dim: 8, lower bound: -43.1455482, upper bound: 43.1455479
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 32.82
Output dim: 8, lower bound: -43.1406803, upper bound: 43.1406685
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 32.82
Output dim: 8, lower bound: -43.1406803, upper bound: 43.1406685
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 32.82
Output dim: 8, lower bound: -43.1297703, upper bound: 43.1297648
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 32.82
Output dim: 8, lower bound: -43.1297703, upper bound: 43.1297648
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 32.82
Output dim: 8, lower bound: -43.1337004, upper bound: 43.1337115
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 32.82
Output dim: 8, lower bound: -43.1337004, upper bound: 43.1337115
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 32.82
Output dim: 8, lower bound: -43.1324350, upper bound: 43.1324354
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 32.82
Output dim: 8, lower bound: -43.1324350, upper bound: 43.1324354

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

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

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1284804, upper bound: 43.1284993
time: 6.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1284893, upper bound: 43.1284902
time: 7.08 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

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

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 216

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1285070, upper bound: 43.1284791
time: 15.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1285070, upper bound: 43.1284817
time: 6.48 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

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

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1270544, upper bound: 43.1270016
time: 12.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1270544, upper bound: 43.1270016
time: 6.27 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1256516, upper bound: 43.1255910
time: 6.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1256826, upper bound: 43.1255681
time: 16.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

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

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1253611, upper bound: 43.1254450
time: 10.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1253611, upper bound: 43.1254450
time: 10.53 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 22.59 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 22.59
Output dim: 8, lower bound: -43.1284804, upper bound: 43.1284993
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 22.59
Output dim: 8, lower bound: -43.1284893, upper bound: 43.1284902
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 22.59
Output dim: 8, lower bound: -43.1285070, upper bound: 43.1284791
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 22.59
Output dim: 8, lower bound: -43.1285070, upper bound: 43.1284817
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 22.59
Output dim: 8, lower bound: -43.1270544, upper bound: 43.1270016
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 22.59
Output dim: 8, lower bound: -43.1270544, upper bound: 43.1270016
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 22.59
Output dim: 8, lower bound: -43.1256516, upper bound: 43.1255910
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 22.59
Output dim: 8, lower bound: -43.1256826, upper bound: 43.1255681
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 22.59
Output dim: 8, lower bound: -43.1253611, upper bound: 43.1254450
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 22.59
Output dim: 8, lower bound: -43.1253611, upper bound: 43.1254450
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.59
Output dim: 8, lower bound: -43.1318268, upper bound: 43.1318979
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.59
Output dim: 8, lower bound: -43.1293805, upper bound: 43.1293369
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.59
Output dim: 8, lower bound: -43.1293501, upper bound: 43.1293657
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.59
Output dim: 8, lower bound: -43.1293626, upper bound: 43.1293805
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.59
Output dim: 8, lower bound: -43.1293657, upper bound: 43.1293794
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.59
Output dim: 8, lower bound: -43.1358385, upper bound: 43.1358550
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.59
Output dim: 8, lower bound: -43.1358563, upper bound: 43.1358359
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.59
Output dim: 8, lower bound: -43.1366917, upper bound: 43.1366936
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.59
Output dim: 8, lower bound: -43.1366929, upper bound: 43.1366918
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.59
Output dim: 8, lower bound: -43.1557949, upper bound: 43.1557917
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.59
Output dim: 8, lower bound: -43.1557949, upper bound: 43.1557917
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.59
Output dim: 8, lower bound: -43.1455482, upper bound: 43.1455479
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.59
Output dim: 8, lower bound: -43.1455482, upper bound: 43.1455479
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.59
Output dim: 8, lower bound: -43.1406803, upper bound: 43.1406685
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.59
Output dim: 8, lower bound: -43.1406803, upper bound: 43.1406685
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.59
Output dim: 8, lower bound: -43.1297703, upper bound: 43.1297648
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.59
Output dim: 8, lower bound: -43.1297703, upper bound: 43.1297648
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.59
Output dim: 8, lower bound: -43.1337004, upper bound: 43.1337115
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.59
Output dim: 8, lower bound: -43.1337004, upper bound: 43.1337115
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.59
Output dim: 8, lower bound: -43.1324350, upper bound: 43.1324354
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.59
Output dim: 8, lower bound: -43.1324350, upper bound: 43.1324354
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=51.530487060546875
rel_dist={8: [-43.17090795605019, 43.17090794479026]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1647236, upper bound: 43.1647231
time: 17.93 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1647236, upper bound: 43.1647231
time: 17.07 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 35.02 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 35.02
Output dim: 8, lower bound: -43.1647236, upper bound: 43.1647231
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 35.02
Output dim: 8, lower bound: -43.1647236, upper bound: 43.1647231

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
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1639970, upper bound: 43.1639970
time: 6.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1639970, upper bound: 43.1639970
time: 7.46 seconds

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

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1638348, upper bound: 43.1638411
time: 7.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1638419, upper bound: 43.1638340
time: 6.46 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 16.96 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 16.96
Output dim: 8, lower bound: -43.1639970, upper bound: 43.1639970
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 16.96
Output dim: 8, lower bound: -43.1639970, upper bound: 43.1639970
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 16.96
Output dim: 8, lower bound: -43.1638348, upper bound: 43.1638411
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 16.96
Output dim: 8, lower bound: -43.1638419, upper bound: 43.1638340

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

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1573812, upper bound: 43.1573812
time: 5.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1573812, upper bound: 43.1573955
time: 6.34 seconds

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

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1639970, upper bound: 43.1639960
time: 6.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1639960, upper bound: 43.1639970
time: 6.65 seconds

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

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1604248, upper bound: 43.1604151
time: 8.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1604141, upper bound: 43.1604253
time: 9.69 seconds

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

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1613969, upper bound: 43.1613981
time: 6.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1614030, upper bound: 43.1613871
time: 7.15 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 14.51 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 14.51
Output dim: 8, lower bound: -43.1573812, upper bound: 43.1573812
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 14.51
Output dim: 8, lower bound: -43.1573812, upper bound: 43.1573955
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 14.51
Output dim: 8, lower bound: -43.1639970, upper bound: 43.1639960
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 14.51
Output dim: 8, lower bound: -43.1639960, upper bound: 43.1639970
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 14.51
Output dim: 8, lower bound: -43.1604248, upper bound: 43.1604151
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 14.51
Output dim: 8, lower bound: -43.1604141, upper bound: 43.1604253
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 14.51
Output dim: 8, lower bound: -43.1613969, upper bound: 43.1613981
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 14.51
Output dim: 8, lower bound: -43.1614030, upper bound: 43.1613871

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

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1561479, upper bound: 43.1561318
time: 6.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1561476, upper bound: 43.1561306
time: 6.06 seconds

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

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1229930, upper bound: 43.1229989
time: 6.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1229930, upper bound: 43.1229989
time: 6.70 seconds

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
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1525148, upper bound: 43.1525169
time: 8.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1525148, upper bound: 43.1525169
time: 6.84 seconds

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

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1473576, upper bound: 43.1473591
time: 5.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1473576, upper bound: 43.1473591
time: 5.48 seconds

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

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1521648, upper bound: 43.1521457
time: 8.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1521499, upper bound: 43.1521672
time: 5.96 seconds

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

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1447497, upper bound: 43.1447542
time: 6.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1447497, upper bound: 43.1447548
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

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1560124, upper bound: 43.1560163
time: 6.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1560124, upper bound: 43.1560163
time: 6.29 seconds

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

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1613545, upper bound: 43.1613435
time: 7.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1613574, upper bound: 43.1613423
time: 7.65 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 22.80 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.80
Output dim: 8, lower bound: -43.1561479, upper bound: 43.1561318
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.80
Output dim: 8, lower bound: -43.1561476, upper bound: 43.1561306
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 22.80
Output dim: 8, lower bound: -43.1229930, upper bound: 43.1229989
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 22.80
Output dim: 8, lower bound: -43.1229930, upper bound: 43.1229989
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.80
Output dim: 8, lower bound: -43.1525148, upper bound: 43.1525169
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.80
Output dim: 8, lower bound: -43.1525148, upper bound: 43.1525169
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.80
Output dim: 8, lower bound: -43.1473576, upper bound: 43.1473591
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.80
Output dim: 8, lower bound: -43.1473576, upper bound: 43.1473591
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.80
Output dim: 8, lower bound: -43.1521648, upper bound: 43.1521457
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.80
Output dim: 8, lower bound: -43.1521499, upper bound: 43.1521672
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.80
Output dim: 8, lower bound: -43.1447497, upper bound: 43.1447542
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.80
Output dim: 8, lower bound: -43.1447497, upper bound: 43.1447548
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.80
Output dim: 8, lower bound: -43.1560124, upper bound: 43.1560163
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.80
Output dim: 8, lower bound: -43.1560124, upper bound: 43.1560163
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.80
Output dim: 8, lower bound: -43.1613545, upper bound: 43.1613435
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.80
Output dim: 8, lower bound: -43.1613574, upper bound: 43.1613423

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

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 119

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1333256, upper bound: 43.1333163
time: 6.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1333256, upper bound: 43.1333163
time: 7.03 seconds

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

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1288441, upper bound: 43.1288448
time: 7.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1288441, upper bound: 43.1288448
time: 6.95 seconds

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
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 153

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1525130, upper bound: 43.1525169
time: 7.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1525145, upper bound: 43.1525143
time: 24.26 seconds

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

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1525097, upper bound: 43.1525118
time: 6.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1525096, upper bound: 43.1525165
time: 7.77 seconds

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
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 117

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1468422, upper bound: 43.1468465
time: 6.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1468470, upper bound: 43.1468417
time: 5.58 seconds

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

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1338253, upper bound: 43.1338236
time: 23.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1338253, upper bound: 43.1338236
time: 57.32 seconds

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

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1521648, upper bound: 43.1521444
time: 7.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1521629, upper bound: 43.1521457
time: 8.22 seconds

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

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1230475, upper bound: 43.1230635
time: 6.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1230475, upper bound: 43.1230635
time: 6.10 seconds

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

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1298302, upper bound: 43.1298308
time: 6.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1298302, upper bound: 43.1298308
time: 6.62 seconds

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

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1447486, upper bound: 43.1447372
time: 4.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1447348, upper bound: 43.1447526
time: 5.69 seconds

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

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1483192, upper bound: 43.1483104
time: 5.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1483174, upper bound: 43.1483151
time: 6.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1518195, upper bound: 43.1518213
time: 6.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1518195, upper bound: 43.1518213
time: 6.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 117

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1292207, upper bound: 43.1292186
time: 7.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1292170, upper bound: 43.1292186
time: 5.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1602672, upper bound: 43.1602430
time: 6.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1602616, upper bound: 43.1602422
time: 12.98 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 23.25 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 8, lower bound: -43.1333256, upper bound: 43.1333163
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 8, lower bound: -43.1333256, upper bound: 43.1333163
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 8, lower bound: -43.1288441, upper bound: 43.1288448
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 8, lower bound: -43.1288441, upper bound: 43.1288448
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 8, lower bound: -43.1525130, upper bound: 43.1525169
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 8, lower bound: -43.1525145, upper bound: 43.1525143
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 8, lower bound: -43.1525097, upper bound: 43.1525118
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 8, lower bound: -43.1525096, upper bound: 43.1525165
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 8, lower bound: -43.1468422, upper bound: 43.1468465
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 8, lower bound: -43.1468470, upper bound: 43.1468417
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 8, lower bound: -43.1338253, upper bound: 43.1338236
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 8, lower bound: -43.1338253, upper bound: 43.1338236
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 8, lower bound: -43.1521648, upper bound: 43.1521444
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 8, lower bound: -43.1521629, upper bound: 43.1521457
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 23.25
Output dim: 8, lower bound: -43.1230475, upper bound: 43.1230635
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 23.25
Output dim: 8, lower bound: -43.1230475, upper bound: 43.1230635
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 8, lower bound: -43.1298302, upper bound: 43.1298308
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 8, lower bound: -43.1298302, upper bound: 43.1298308
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 8, lower bound: -43.1447486, upper bound: 43.1447372
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 8, lower bound: -43.1447348, upper bound: 43.1447526
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 8, lower bound: -43.1483192, upper bound: 43.1483104
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 8, lower bound: -43.1483174, upper bound: 43.1483151
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 8, lower bound: -43.1518195, upper bound: 43.1518213
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 8, lower bound: -43.1518195, upper bound: 43.1518213
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 8, lower bound: -43.1292207, upper bound: 43.1292186
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 8, lower bound: -43.1292170, upper bound: 43.1292186
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 8, lower bound: -43.1602672, upper bound: 43.1602430
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 8, lower bound: -43.1602616, upper bound: 43.1602422

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1333256, upper bound: 43.1333163
time: 3.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1333255, upper bound: 43.1333159
time: 5.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 50

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 216

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1333228, upper bound: 43.1333163
time: 7.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1333256, upper bound: 43.1333141
time: 5.88 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 14.31 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.31
Output dim: 8, lower bound: -43.1333256, upper bound: 43.1333163
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.31
Output dim: 8, lower bound: -43.1333255, upper bound: 43.1333159
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.31
Output dim: 8, lower bound: -43.1333228, upper bound: 43.1333163
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.31
Output dim: 8, lower bound: -43.1333256, upper bound: 43.1333141
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.31
Output dim: 8, lower bound: -43.1288441, upper bound: 43.1288448
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.31
Output dim: 8, lower bound: -43.1288441, upper bound: 43.1288448
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.31
Output dim: 8, lower bound: -43.1525130, upper bound: 43.1525169
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.31
Output dim: 8, lower bound: -43.1525145, upper bound: 43.1525143
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.31
Output dim: 8, lower bound: -43.1525097, upper bound: 43.1525118
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.31
Output dim: 8, lower bound: -43.1525096, upper bound: 43.1525165
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.31
Output dim: 8, lower bound: -43.1468422, upper bound: 43.1468465
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.31
Output dim: 8, lower bound: -43.1468470, upper bound: 43.1468417
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.31
Output dim: 8, lower bound: -43.1338253, upper bound: 43.1338236
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.31
Output dim: 8, lower bound: -43.1338253, upper bound: 43.1338236
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.31
Output dim: 8, lower bound: -43.1521648, upper bound: 43.1521444
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.31
Output dim: 8, lower bound: -43.1521629, upper bound: 43.1521457
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.31
Output dim: 8, lower bound: -43.1298302, upper bound: 43.1298308
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.31
Output dim: 8, lower bound: -43.1298302, upper bound: 43.1298308
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.31
Output dim: 8, lower bound: -43.1447486, upper bound: 43.1447372
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.31
Output dim: 8, lower bound: -43.1447348, upper bound: 43.1447526
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.31
Output dim: 8, lower bound: -43.1483192, upper bound: 43.1483104
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.31
Output dim: 8, lower bound: -43.1483174, upper bound: 43.1483151
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.31
Output dim: 8, lower bound: -43.1518195, upper bound: 43.1518213
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.31
Output dim: 8, lower bound: -43.1518195, upper bound: 43.1518213
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.31
Output dim: 8, lower bound: -43.1292207, upper bound: 43.1292186
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.31
Output dim: 8, lower bound: -43.1292170, upper bound: 43.1292186
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.31
Output dim: 8, lower bound: -43.1602672, upper bound: 43.1602430
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.31
Output dim: 8, lower bound: -43.1602616, upper bound: 43.1602422
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=51.530487060546875
rel_dist={8: [-43.17074367834749, 43.17074366907076]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1685515, upper bound: 43.1685525
time: 6.01 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1685515, upper bound: 43.1685515
time: 6.78 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 12.81 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 12.81
Output dim: 8, lower bound: -43.1685515, upper bound: 43.1685525
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 12.81
Output dim: 8, lower bound: -43.1685515, upper bound: 43.1685515

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

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 216

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1684745, upper bound: 43.1684756
time: 7.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1684745, upper bound: 43.1684756
time: 7.56 seconds

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

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 216

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1632906, upper bound: 43.1632897
time: 5.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1632906, upper bound: 43.1632903
time: 7.32 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 14.46 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 14.46
Output dim: 8, lower bound: -43.1684745, upper bound: 43.1684756
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 14.46
Output dim: 8, lower bound: -43.1684745, upper bound: 43.1684756
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 14.46
Output dim: 8, lower bound: -43.1632906, upper bound: 43.1632897
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 14.46
Output dim: 8, lower bound: -43.1632906, upper bound: 43.1632903

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

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1599248, upper bound: 43.1599248
time: 7.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1599248, upper bound: 43.1599248
time: 6.99 seconds

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
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 50

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1622847, upper bound: 43.1622847
time: 6.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1622847, upper bound: 43.1622988
time: 8.41 seconds

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

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1581732, upper bound: 43.1581834
time: 6.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1581732, upper bound: 43.1581729
time: 7.34 seconds

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

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1632529, upper bound: 43.1632531
time: 7.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1632529, upper bound: 43.1632531
time: 7.14 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 15.90 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 15.90
Output dim: 8, lower bound: -43.1599248, upper bound: 43.1599248
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 15.90
Output dim: 8, lower bound: -43.1599248, upper bound: 43.1599248
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 15.90
Output dim: 8, lower bound: -43.1622847, upper bound: 43.1622847
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 15.90
Output dim: 8, lower bound: -43.1622847, upper bound: 43.1622988
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 15.90
Output dim: 8, lower bound: -43.1581732, upper bound: 43.1581834
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 15.90
Output dim: 8, lower bound: -43.1581732, upper bound: 43.1581729
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 15.90
Output dim: 8, lower bound: -43.1632529, upper bound: 43.1632531
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 15.90
Output dim: 8, lower bound: -43.1632529, upper bound: 43.1632531

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

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1543824, upper bound: 43.1543856
time: 8.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1543824, upper bound: 43.1543824
time: 7.16 seconds

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
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1599226, upper bound: 43.1599226
time: 8.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1599226, upper bound: 43.1599248
time: 8.39 seconds

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

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1563855, upper bound: 43.1563855
time: 6.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1563855, upper bound: 43.1563855
time: 8.41 seconds

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

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1622755, upper bound: 43.1622988
time: 6.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1622859, upper bound: 43.1622922
time: 21.07 seconds

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
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 119

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1406080, upper bound: 43.1406098
time: 10.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1406080, upper bound: 43.1406098
time: 9.17 seconds

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

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1500560, upper bound: 43.1500491
time: 7.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1500491, upper bound: 43.1500491
time: 7.38 seconds

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
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 216

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1632321, upper bound: 43.1632316
time: 7.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1632321, upper bound: 43.1632324
time: 12.02 seconds

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

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1629679, upper bound: 43.1629672
time: 7.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1629679, upper bound: 43.1629672
time: 6.69 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 15.37 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.37
Output dim: 8, lower bound: -43.1543824, upper bound: 43.1543856
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.37
Output dim: 8, lower bound: -43.1543824, upper bound: 43.1543824
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.37
Output dim: 8, lower bound: -43.1599226, upper bound: 43.1599226
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.37
Output dim: 8, lower bound: -43.1599226, upper bound: 43.1599248
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.37
Output dim: 8, lower bound: -43.1563855, upper bound: 43.1563855
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.37
Output dim: 8, lower bound: -43.1563855, upper bound: 43.1563855
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.37
Output dim: 8, lower bound: -43.1622755, upper bound: 43.1622988
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.37
Output dim: 8, lower bound: -43.1622859, upper bound: 43.1622922
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.37
Output dim: 8, lower bound: -43.1406080, upper bound: 43.1406098
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.37
Output dim: 8, lower bound: -43.1406080, upper bound: 43.1406098
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.37
Output dim: 8, lower bound: -43.1500560, upper bound: 43.1500491
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.37
Output dim: 8, lower bound: -43.1500491, upper bound: 43.1500491
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.37
Output dim: 8, lower bound: -43.1632321, upper bound: 43.1632316
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.37
Output dim: 8, lower bound: -43.1632321, upper bound: 43.1632324
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.37
Output dim: 8, lower bound: -43.1629679, upper bound: 43.1629672
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.37
Output dim: 8, lower bound: -43.1629679, upper bound: 43.1629672

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

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1121587, upper bound: 43.1121606
time: 23.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1121587, upper bound: 43.1121606
time: 14.16 seconds

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

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1227361, upper bound: 43.1227362
time: 7.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1227396, upper bound: 43.1227362
time: 13.70 seconds

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

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1498320, upper bound: 43.1498313
time: 5.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1498326, upper bound: 43.1498313
time: 7.39 seconds

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
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 117

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1453818, upper bound: 43.1453836
time: 8.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1453818, upper bound: 43.1453836
time: 6.63 seconds

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

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1332870, upper bound: 43.1332850
time: 6.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1332870, upper bound: 43.1332850
time: 6.25 seconds

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

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1461638, upper bound: 43.1461637
time: 6.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1461681, upper bound: 43.1461637
time: 7.41 seconds

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

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 117

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1192097, upper bound: 43.1192120
time: 6.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -43.1192096, upper bound: 43.1192120
time: 10.33 seconds

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

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1386299, upper bound: 43.1386376
time: 6.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1386299, upper bound: 43.1386376
time: 7.29 seconds

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

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 175

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1323086, upper bound: 43.1323118
time: 6.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1323086, upper bound: 43.1323118
time: 7.03 seconds

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

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1405998, upper bound: 43.1406098
time: 6.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1406080, upper bound: 43.1406011
time: 7.64 seconds

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

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1428986, upper bound: 43.1428986
time: 7.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1429020, upper bound: 43.1428986
time: 14.72 seconds

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

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1300870, upper bound: 43.1300870
time: 7.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1300870, upper bound: 43.1300870
time: 7.96 seconds

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

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1632321, upper bound: 43.1632298
time: 7.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1632306, upper bound: 43.1632324
time: 21.45 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

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

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1566034, upper bound: 43.1566035
time: 8.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1566034, upper bound: 43.1566045
time: 7.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

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

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 119

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1388670, upper bound: 43.1388675
time: 7.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1388670, upper bound: 43.1388675
time: 7.21 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

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

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1564758, upper bound: 43.1564777
time: 7.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1564760, upper bound: 43.1564777
time: 7.21 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 16.28 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.28
Output dim: 8, lower bound: -43.1121587, upper bound: 43.1121606
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.28
Output dim: 8, lower bound: -43.1121587, upper bound: 43.1121606
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.28
Output dim: 8, lower bound: -43.1227361, upper bound: 43.1227362
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.28
Output dim: 8, lower bound: -43.1227396, upper bound: 43.1227362
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.28
Output dim: 8, lower bound: -43.1498320, upper bound: 43.1498313
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.28
Output dim: 8, lower bound: -43.1498326, upper bound: 43.1498313
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.28
Output dim: 8, lower bound: -43.1453818, upper bound: 43.1453836
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.28
Output dim: 8, lower bound: -43.1453818, upper bound: 43.1453836
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.28
Output dim: 8, lower bound: -43.1332870, upper bound: 43.1332850
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.28
Output dim: 8, lower bound: -43.1332870, upper bound: 43.1332850
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.28
Output dim: 8, lower bound: -43.1461638, upper bound: 43.1461637
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.28
Output dim: 8, lower bound: -43.1461681, upper bound: 43.1461637
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.28
Output dim: 8, lower bound: -43.1192097, upper bound: 43.1192120
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.28
Output dim: 8, lower bound: -43.1192096, upper bound: 43.1192120
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.28
Output dim: 8, lower bound: -43.1386299, upper bound: 43.1386376
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.28
Output dim: 8, lower bound: -43.1386299, upper bound: 43.1386376
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.28
Output dim: 8, lower bound: -43.1323086, upper bound: 43.1323118
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.28
Output dim: 8, lower bound: -43.1323086, upper bound: 43.1323118
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.28
Output dim: 8, lower bound: -43.1405998, upper bound: 43.1406098
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.28
Output dim: 8, lower bound: -43.1406080, upper bound: 43.1406011
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.28
Output dim: 8, lower bound: -43.1428986, upper bound: 43.1428986
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.28
Output dim: 8, lower bound: -43.1429020, upper bound: 43.1428986
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.28
Output dim: 8, lower bound: -43.1300870, upper bound: 43.1300870
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.28
Output dim: 8, lower bound: -43.1300870, upper bound: 43.1300870
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.28
Output dim: 8, lower bound: -43.1632321, upper bound: 43.1632298
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.28
Output dim: 8, lower bound: -43.1632306, upper bound: 43.1632324
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.28
Output dim: 8, lower bound: -43.1566034, upper bound: 43.1566035
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.28
Output dim: 8, lower bound: -43.1566034, upper bound: 43.1566045
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.28
Output dim: 8, lower bound: -43.1388670, upper bound: 43.1388675
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.28
Output dim: 8, lower bound: -43.1388670, upper bound: 43.1388675
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.28
Output dim: 8, lower bound: -43.1564758, upper bound: 43.1564777
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.28
Output dim: 8, lower bound: -43.1564760, upper bound: 43.1564777

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1445057, upper bound: 43.1445031
time: 7.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1445057, upper bound: 43.1445030
time: 6.61 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 17.90 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 17.90
Output dim: 8, lower bound: -43.1445057, upper bound: 43.1445031
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 17.90
Output dim: 8, lower bound: -43.1445057, upper bound: 43.1445030
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.90
Output dim: 8, lower bound: -43.1498326, upper bound: 43.1498313
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.90
Output dim: 8, lower bound: -43.1453818, upper bound: 43.1453836
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.90
Output dim: 8, lower bound: -43.1453818, upper bound: 43.1453836
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.90
Output dim: 8, lower bound: -43.1332870, upper bound: 43.1332850
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.90
Output dim: 8, lower bound: -43.1332870, upper bound: 43.1332850
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.90
Output dim: 8, lower bound: -43.1461638, upper bound: 43.1461637
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.90
Output dim: 8, lower bound: -43.1461681, upper bound: 43.1461637
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.90
Output dim: 8, lower bound: -43.1386299, upper bound: 43.1386376
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.90
Output dim: 8, lower bound: -43.1386299, upper bound: 43.1386376
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.90
Output dim: 8, lower bound: -43.1323086, upper bound: 43.1323118
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.90
Output dim: 8, lower bound: -43.1323086, upper bound: 43.1323118
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.90
Output dim: 8, lower bound: -43.1405998, upper bound: 43.1406098
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.90
Output dim: 8, lower bound: -43.1406080, upper bound: 43.1406011
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.90
Output dim: 8, lower bound: -43.1428986, upper bound: 43.1428986
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.90
Output dim: 8, lower bound: -43.1429020, upper bound: 43.1428986
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.90
Output dim: 8, lower bound: -43.1300870, upper bound: 43.1300870
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.90
Output dim: 8, lower bound: -43.1300870, upper bound: 43.1300870
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.90
Output dim: 8, lower bound: -43.1632321, upper bound: 43.1632298
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.90
Output dim: 8, lower bound: -43.1632306, upper bound: 43.1632324
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.90
Output dim: 8, lower bound: -43.1566034, upper bound: 43.1566035
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.90
Output dim: 8, lower bound: -43.1566034, upper bound: 43.1566045
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.90
Output dim: 8, lower bound: -43.1388670, upper bound: 43.1388675
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.90
Output dim: 8, lower bound: -43.1388670, upper bound: 43.1388675
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.90
Output dim: 8, lower bound: -43.1564758, upper bound: 43.1564777
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.90
Output dim: 8, lower bound: -43.1564760, upper bound: 43.1564777
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=51.530487060546875
rel_dist={8: [-43.1705810903977, 43.17058109163787]}

## Binary Search with RS_random_Z Result
status: None
Maximum delta epsilon: None
execution time: 1822.68 seconds
