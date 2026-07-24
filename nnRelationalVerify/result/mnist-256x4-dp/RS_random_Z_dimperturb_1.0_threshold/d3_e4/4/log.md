## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 10.310653145


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=78, inp2_unstable=78, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540)
1: (-4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358)
2: (-5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447)
3: (-6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520)
4: (-6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002)
5: (-5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936)
6: (-4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792)
7: (-5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823)
8: (-7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416)
9: (-4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.45 + 5.41 = 6.87 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -10.8533189, upper bound: 10.8533189

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8513263, upper bound: 10.8513263
time: 3.37 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8513263, upper bound: 10.8513263
time: 3.14 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 6.53 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 6.53
Output dim: 8, lower bound: -10.8513263, upper bound: 10.8513263
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 6.53
Output dim: 8, lower bound: -10.8513263, upper bound: 10.8513263

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=78, inp2_unstable=78, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 218

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8506146, upper bound: 10.8506148
time: 4.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8506145, upper bound: 10.8506145
time: 1.92 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=78, inp2_unstable=78, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8506146, upper bound: 10.8506140
time: 2.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8506145, upper bound: 10.8506145
time: 1.92 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 7.70 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 7.70
Output dim: 8, lower bound: -10.8506146, upper bound: 10.8506148
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 7.70
Output dim: 8, lower bound: -10.8506145, upper bound: 10.8506145
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 7.70
Output dim: 8, lower bound: -10.8506146, upper bound: 10.8506140
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 7.70
Output dim: 8, lower bound: -10.8506145, upper bound: 10.8506145

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=78, inp2_unstable=78, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8500409, upper bound: 10.8500409
time: 10.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8500409, upper bound: 10.8500403
time: 10.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=78, inp2_unstable=78, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8501789, upper bound: 10.8501790
time: 3.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8501789, upper bound: 10.8501781
time: 2.12 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=78, inp2_unstable=78, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8480215, upper bound: 10.8480217
time: 3.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8480215, upper bound: 10.8480216
time: 5.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=78, inp2_unstable=78, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8492329, upper bound: 10.8492329
time: 4.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8492329, upper bound: 10.8492329
time: 4.46 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 9.77 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 9.77
Output dim: 8, lower bound: -10.8500409, upper bound: 10.8500409
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 9.77
Output dim: 8, lower bound: -10.8500409, upper bound: 10.8500403
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 9.77
Output dim: 8, lower bound: -10.8501789, upper bound: 10.8501790
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 9.77
Output dim: 8, lower bound: -10.8501789, upper bound: 10.8501781
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 9.77
Output dim: 8, lower bound: -10.8480215, upper bound: 10.8480217
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 9.77
Output dim: 8, lower bound: -10.8480215, upper bound: 10.8480216
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 9.77
Output dim: 8, lower bound: -10.8492329, upper bound: 10.8492329
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 9.77
Output dim: 8, lower bound: -10.8492329, upper bound: 10.8492329

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=78, inp2_unstable=78, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8500409, upper bound: 10.8500400
time: 1.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8500409, upper bound: 10.8500408
time: 2.42 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=78, inp2_unstable=78, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 218

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8461157, upper bound: 10.8461165
time: 3.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8461157, upper bound: 10.8461165
time: 3.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=78, inp2_unstable=78, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5817370, upper bound: 10.5817370
time: 1.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5817370, upper bound: 10.5817370
time: 1.44 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=78, inp2_unstable=78, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8488936, upper bound: 10.8488936
time: 3.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8488936, upper bound: 10.8488935
time: 1.93 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=78, inp2_unstable=78, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8475239, upper bound: 10.8475242
time: 1.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8475237, upper bound: 10.8475243
time: 2.44 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=78, inp2_unstable=78, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5467561, upper bound: 10.5467561
time: 1.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5467561, upper bound: 10.5467561
time: 1.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=78, inp2_unstable=78, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8488158, upper bound: 10.8488159
time: 4.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8488159, upper bound: 10.8488158
time: 3.11 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=78, inp2_unstable=78, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8453543, upper bound: 10.8453545
time: 1.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8453543, upper bound: 10.8453545
time: 1.45 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 6.32 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.32
Output dim: 8, lower bound: -10.8500409, upper bound: 10.8500400
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.32
Output dim: 8, lower bound: -10.8500409, upper bound: 10.8500408
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.32
Output dim: 8, lower bound: -10.8461157, upper bound: 10.8461165
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.32
Output dim: 8, lower bound: -10.8461157, upper bound: 10.8461165
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.32
Output dim: 8, lower bound: -10.5817370, upper bound: 10.5817370
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.32
Output dim: 8, lower bound: -10.5817370, upper bound: 10.5817370
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.32
Output dim: 8, lower bound: -10.8488936, upper bound: 10.8488936
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.32
Output dim: 8, lower bound: -10.8488936, upper bound: 10.8488935
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.32
Output dim: 8, lower bound: -10.8475239, upper bound: 10.8475242
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.32
Output dim: 8, lower bound: -10.8475237, upper bound: 10.8475243
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.32
Output dim: 8, lower bound: -10.5467561, upper bound: 10.5467561
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.32
Output dim: 8, lower bound: -10.5467561, upper bound: 10.5467561
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.32
Output dim: 8, lower bound: -10.8488158, upper bound: 10.8488159
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.32
Output dim: 8, lower bound: -10.8488159, upper bound: 10.8488158
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.32
Output dim: 8, lower bound: -10.8453543, upper bound: 10.8453545
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.32
Output dim: 8, lower bound: -10.8453543, upper bound: 10.8453545

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=78, inp2_unstable=78, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8453481, upper bound: 10.8453502
time: 1.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8453481, upper bound: 10.8453503
time: 2.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=78, inp2_unstable=78, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8494857, upper bound: 10.8494857
time: 3.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8494857, upper bound: 10.8494855
time: 4.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=78, inp2_unstable=78, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8461157, upper bound: 10.8461165
time: 7.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8461157, upper bound: 10.8461165
time: 2.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=78, inp2_unstable=78, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8390995, upper bound: 10.8390996
time: 1.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8390995, upper bound: 10.8390996
time: 1.42 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=78, inp2_unstable=78, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 218

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.3082535, upper bound: 10.3082535
time: 1.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.3082535, upper bound: 10.3082535
time: 1.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=78, inp2_unstable=78, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.3082535, upper bound: 10.3082535
time: 1.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.3082535, upper bound: 10.3082535
time: 1.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=78, inp2_unstable=78, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8460262, upper bound: 10.8460262
time: 6.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8460262, upper bound: 10.8460262
time: 2.03 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=78, inp2_unstable=78, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8168819, upper bound: 10.8168820
time: 2.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8168819, upper bound: 10.8168820
time: 2.19 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=78, inp2_unstable=78, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6825183, upper bound: 10.6825183
time: 1.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6825183, upper bound: 10.6825183
time: 1.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=78, inp2_unstable=78, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6825183, upper bound: 10.6825183
time: 1.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6825183, upper bound: 10.6825183
time: 2.14 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=78, inp2_unstable=78, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.3064763, upper bound: 10.3064763
time: 1.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.3064763, upper bound: 10.3064763
time: 1.98 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=78, inp2_unstable=78, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 218

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3363539, upper bound: 10.3363539
time: 1.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3363539, upper bound: 10.3363539
time: 1.25 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=78, inp2_unstable=78, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8478890, upper bound: 10.8478890
time: 1.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8478890, upper bound: 10.8478890
time: 4.55 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=78, inp2_unstable=78, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8467233, upper bound: 10.8467286
time: 1.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8467281, upper bound: 10.8467241
time: 8.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=78, inp2_unstable=78, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 218

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8322147, upper bound: 10.8322157
time: 1.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8322142, upper bound: 10.8322162
time: 1.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=78, inp2_unstable=78, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4438576, upper bound: 10.4438576
time: 1.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4438576, upper bound: 10.4438576
time: 1.31 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 6.02 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.02
Output dim: 8, lower bound: -10.8453481, upper bound: 10.8453502
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.02
Output dim: 8, lower bound: -10.8453481, upper bound: 10.8453503
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.02
Output dim: 8, lower bound: -10.8494857, upper bound: 10.8494857
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.02
Output dim: 8, lower bound: -10.8494857, upper bound: 10.8494855
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.02
Output dim: 8, lower bound: -10.8461157, upper bound: 10.8461165
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.02
Output dim: 8, lower bound: -10.8461157, upper bound: 10.8461165
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.02
Output dim: 8, lower bound: -10.8390995, upper bound: 10.8390996
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.02
Output dim: 8, lower bound: -10.8390995, upper bound: 10.8390996
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 6.02
Output dim: 8, lower bound: -10.3082535, upper bound: 10.3082535
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 6.02
Output dim: 8, lower bound: -10.3082535, upper bound: 10.3082535
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 6.02
Output dim: 8, lower bound: -10.3082535, upper bound: 10.3082535
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 6.02
Output dim: 8, lower bound: -10.3082535, upper bound: 10.3082535
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.02
Output dim: 8, lower bound: -10.8460262, upper bound: 10.8460262
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.02
Output dim: 8, lower bound: -10.8460262, upper bound: 10.8460262
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.02
Output dim: 8, lower bound: -10.8168819, upper bound: 10.8168820
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.02
Output dim: 8, lower bound: -10.8168819, upper bound: 10.8168820
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.02
Output dim: 8, lower bound: -10.6825183, upper bound: 10.6825183
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.02
Output dim: 8, lower bound: -10.6825183, upper bound: 10.6825183
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.02
Output dim: 8, lower bound: -10.6825183, upper bound: 10.6825183
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.02
Output dim: 8, lower bound: -10.6825183, upper bound: 10.6825183
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 6.02
Output dim: 8, lower bound: -10.3064763, upper bound: 10.3064763
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 6.02
Output dim: 8, lower bound: -10.3064763, upper bound: 10.3064763
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.02
Output dim: 8, lower bound: -10.3363539, upper bound: 10.3363539
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.02
Output dim: 8, lower bound: -10.3363539, upper bound: 10.3363539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.02
Output dim: 8, lower bound: -10.8478890, upper bound: 10.8478890
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.02
Output dim: 8, lower bound: -10.8478890, upper bound: 10.8478890
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.02
Output dim: 8, lower bound: -10.8467233, upper bound: 10.8467286
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.02
Output dim: 8, lower bound: -10.8467281, upper bound: 10.8467241
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.02
Output dim: 8, lower bound: -10.8322147, upper bound: 10.8322157
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.02
Output dim: 8, lower bound: -10.8322142, upper bound: 10.8322162
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.02
Output dim: 8, lower bound: -10.4438576, upper bound: 10.4438576
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.02
Output dim: 8, lower bound: -10.4438576, upper bound: 10.4438576

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=78, inp2_unstable=78, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6339803, upper bound: 10.6339804
time: 1.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6339804, upper bound: 10.6339808
time: 1.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=78, inp2_unstable=78, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 218

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7571991, upper bound: 10.7571992
time: 1.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7571993, upper bound: 10.7571991
time: 1.47 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=78, inp2_unstable=78, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 218

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8470042, upper bound: 10.8470043
time: 2.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8470042, upper bound: 10.8470042
time: 2.43 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=78, inp2_unstable=78, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8494857, upper bound: 10.8494857
time: 4.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8494856, upper bound: 10.8494856
time: 5.41 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=78, inp2_unstable=78, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7694903, upper bound: 10.7694903
time: 2.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7694903, upper bound: 10.7694903
time: 2.50 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=78, inp2_unstable=78, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8450121, upper bound: 10.8450122
time: 1.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8450121, upper bound: 10.8450122
time: 1.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=78, inp2_unstable=78, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7437054, upper bound: 10.7437054
time: 1.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7437054, upper bound: 10.7437054
time: 1.94 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=78, inp2_unstable=78, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7437054, upper bound: 10.7437054
time: 1.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7437054, upper bound: 10.7437054
time: 1.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=78, inp2_unstable=78, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5541862, upper bound: 10.5541874
time: 1.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5541862, upper bound: 10.5541874
time: 1.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=78, inp2_unstable=78, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 218

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6189809, upper bound: 10.6189809
time: 2.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6189809, upper bound: 10.6189809
time: 1.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=78, inp2_unstable=78, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5630472, upper bound: 10.5630465
time: 1.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5630465, upper bound: 10.5630466
time: 1.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=78, inp2_unstable=78, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5381283, upper bound: 10.5381283
time: 1.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5381283, upper bound: 10.5381283
time: 1.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=78, inp2_unstable=78, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4615045, upper bound: 10.4615045
time: 1.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4615045, upper bound: 10.4615045
time: 1.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=78, inp2_unstable=78, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 218

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6620022, upper bound: 10.6620023
time: 1.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6620022, upper bound: 10.6620023
time: 1.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=78, inp2_unstable=78, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6825183, upper bound: 10.6825183
time: 2.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6825183, upper bound: 10.6825183
time: 2.09 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=78, inp2_unstable=78, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 218

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6825183, upper bound: 10.6825183
time: 2.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6825183, upper bound: 10.6825183
time: 2.09 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=78, inp2_unstable=78, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.0827421, upper bound: 10.0827434
time: 1.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.0827421, upper bound: 10.0827425
time: 1.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=78, inp2_unstable=78, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 218

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3363539, upper bound: 10.3363539
time: 1.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3363539, upper bound: 10.3363539
time: 2.06 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=78, inp2_unstable=78, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8478890, upper bound: 10.8478890
time: 1.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8478890, upper bound: 10.8478890
time: 2.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=78, inp2_unstable=78, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8469460, upper bound: 10.8469460
time: 2.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8469460, upper bound: 10.8469460
time: 2.22 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=78, inp2_unstable=78, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8443905, upper bound: 10.8443921
time: 1.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8443905, upper bound: 10.8443921
time: 1.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=78, inp2_unstable=78, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6787946, upper bound: 10.6787948
time: 1.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6787946, upper bound: 10.6787948
time: 1.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=78, inp2_unstable=78, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 218

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5760306, upper bound: 10.5760314
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5760306, upper bound: 10.5760314
time: 1.41 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=78, inp2_unstable=78, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 218

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6787946, upper bound: 10.6787948
time: 1.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6787946, upper bound: 10.6787948
time: 1.94 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=78, inp2_unstable=78, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4438576, upper bound: 10.4438576
time: 1.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4438576, upper bound: 10.4438576
time: 1.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=78, inp2_unstable=78, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1389129, upper bound: 10.1389119
time: 1.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1389129, upper bound: 10.1389119
time: 1.55 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 4.48 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 8, lower bound: -10.6339803, upper bound: 10.6339804
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 8, lower bound: -10.6339804, upper bound: 10.6339808
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 8, lower bound: -10.7571991, upper bound: 10.7571992
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 8, lower bound: -10.7571993, upper bound: 10.7571991
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 8, lower bound: -10.8470042, upper bound: 10.8470043
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 8, lower bound: -10.8470042, upper bound: 10.8470042
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 8, lower bound: -10.8494857, upper bound: 10.8494857
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 8, lower bound: -10.8494856, upper bound: 10.8494856
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 8, lower bound: -10.7694903, upper bound: 10.7694903
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 8, lower bound: -10.7694903, upper bound: 10.7694903
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 8, lower bound: -10.8450121, upper bound: 10.8450122
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 8, lower bound: -10.8450121, upper bound: 10.8450122
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 8, lower bound: -10.7437054, upper bound: 10.7437054
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 8, lower bound: -10.7437054, upper bound: 10.7437054
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 8, lower bound: -10.7437054, upper bound: 10.7437054
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 8, lower bound: -10.7437054, upper bound: 10.7437054
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 8, lower bound: -10.5541862, upper bound: 10.5541874
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 8, lower bound: -10.5541862, upper bound: 10.5541874
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 8, lower bound: -10.6189809, upper bound: 10.6189809
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 8, lower bound: -10.6189809, upper bound: 10.6189809
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 8, lower bound: -10.5630472, upper bound: 10.5630465
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 8, lower bound: -10.5630465, upper bound: 10.5630466
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 8, lower bound: -10.5381283, upper bound: 10.5381283
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 8, lower bound: -10.5381283, upper bound: 10.5381283
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 8, lower bound: -10.4615045, upper bound: 10.4615045
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 8, lower bound: -10.4615045, upper bound: 10.4615045
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 8, lower bound: -10.6620022, upper bound: 10.6620023
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 8, lower bound: -10.6620022, upper bound: 10.6620023
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 8, lower bound: -10.6825183, upper bound: 10.6825183
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 8, lower bound: -10.6825183, upper bound: 10.6825183
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 8, lower bound: -10.6825183, upper bound: 10.6825183
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 8, lower bound: -10.6825183, upper bound: 10.6825183
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.48
Output dim: 8, lower bound: -10.0827421, upper bound: 10.0827434
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.48
Output dim: 8, lower bound: -10.0827421, upper bound: 10.0827425
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 8, lower bound: -10.3363539, upper bound: 10.3363539
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 8, lower bound: -10.3363539, upper bound: 10.3363539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 8, lower bound: -10.8478890, upper bound: 10.8478890
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 8, lower bound: -10.8478890, upper bound: 10.8478890
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 8, lower bound: -10.8469460, upper bound: 10.8469460
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 8, lower bound: -10.8469460, upper bound: 10.8469460
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 8, lower bound: -10.8443905, upper bound: 10.8443921
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 8, lower bound: -10.8443905, upper bound: 10.8443921
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 8, lower bound: -10.6787946, upper bound: 10.6787948
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 8, lower bound: -10.6787946, upper bound: 10.6787948
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 8, lower bound: -10.5760306, upper bound: 10.5760314
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 8, lower bound: -10.5760306, upper bound: 10.5760314
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 8, lower bound: -10.6787946, upper bound: 10.6787948
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 8, lower bound: -10.6787946, upper bound: 10.6787948
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 8, lower bound: -10.4438576, upper bound: 10.4438576
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 8, lower bound: -10.4438576, upper bound: 10.4438576
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.48
Output dim: 8, lower bound: -10.1389129, upper bound: 10.1389119
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.48
Output dim: 8, lower bound: -10.1389129, upper bound: 10.1389119

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=78, inp2_unstable=78, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5941268, upper bound: 10.5941251
time: 1.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5941251, upper bound: 10.5941251
time: 1.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=78, inp2_unstable=78, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3945778, upper bound: 10.3945781
time: 1.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3945778, upper bound: 10.3945781
time: 1.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=78, inp2_unstable=78, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7225277, upper bound: 10.7225273
time: 2.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7225277, upper bound: 10.7225273
time: 3.09 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=78, inp2_unstable=78, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4952756, upper bound: 10.4952758
time: 4.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4952756, upper bound: 10.4952758
time: 1.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=78, inp2_unstable=78, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8246543, upper bound: 10.8246542
time: 2.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8246540, upper bound: 10.8246542
time: 2.13 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=78, inp2_unstable=78, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 218

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5364706, upper bound: 10.5364706
time: 1.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5364706, upper bound: 10.5364706
time: 1.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=78, inp2_unstable=78, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8479937, upper bound: 10.8479944
time: 4.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8479940, upper bound: 10.8479942
time: 3.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=78, inp2_unstable=78, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8473734, upper bound: 10.8473733
time: 2.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8473734, upper bound: 10.8473733
time: 3.13 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=78, inp2_unstable=78, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 218

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7694903, upper bound: 10.7694903
time: 1.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7694903, upper bound: 10.7694903
time: 1.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=78, inp2_unstable=78, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7694903, upper bound: 10.7694903
time: 1.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7694903, upper bound: 10.7694903
time: 3.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=78, inp2_unstable=78, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7945148, upper bound: 10.7945146
time: 1.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7945148, upper bound: 10.7945146
time: 1.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=78, inp2_unstable=78, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7041013, upper bound: 10.7041017
time: 3.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7041020, upper bound: 10.7041011
time: 1.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=78, inp2_unstable=78, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7437054, upper bound: 10.7437054
time: 2.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7437054, upper bound: 10.7437054
time: 3.05 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=78, inp2_unstable=78, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4532328, upper bound: 10.4532344
time: 1.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4532328, upper bound: 10.4532344
time: 1.46 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=78, inp2_unstable=78, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7437054, upper bound: 10.7437054
time: 2.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7437054, upper bound: 10.7437054
time: 2.20 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 10.28 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 10.28
Output dim: 8, lower bound: -10.5941268, upper bound: 10.5941251
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 10.28
Output dim: 8, lower bound: -10.5941251, upper bound: 10.5941251
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 10.28
Output dim: 8, lower bound: -10.3945778, upper bound: 10.3945781
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 10.28
Output dim: 8, lower bound: -10.3945778, upper bound: 10.3945781
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 10.28
Output dim: 8, lower bound: -10.7225277, upper bound: 10.7225273
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 10.28
Output dim: 8, lower bound: -10.7225277, upper bound: 10.7225273
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 10.28
Output dim: 8, lower bound: -10.4952756, upper bound: 10.4952758
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 10.28
Output dim: 8, lower bound: -10.4952756, upper bound: 10.4952758
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 10.28
Output dim: 8, lower bound: -10.8246543, upper bound: 10.8246542
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 10.28
Output dim: 8, lower bound: -10.8246540, upper bound: 10.8246542
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 10.28
Output dim: 8, lower bound: -10.5364706, upper bound: 10.5364706
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 10.28
Output dim: 8, lower bound: -10.5364706, upper bound: 10.5364706
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 10.28
Output dim: 8, lower bound: -10.8479937, upper bound: 10.8479944
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 10.28
Output dim: 8, lower bound: -10.8479940, upper bound: 10.8479942
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 10.28
Output dim: 8, lower bound: -10.8473734, upper bound: 10.8473733
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 10.28
Output dim: 8, lower bound: -10.8473734, upper bound: 10.8473733
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 10.28
Output dim: 8, lower bound: -10.7694903, upper bound: 10.7694903
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 10.28
Output dim: 8, lower bound: -10.7694903, upper bound: 10.7694903
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 10.28
Output dim: 8, lower bound: -10.7694903, upper bound: 10.7694903
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 10.28
Output dim: 8, lower bound: -10.7694903, upper bound: 10.7694903
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 10.28
Output dim: 8, lower bound: -10.7945148, upper bound: 10.7945146
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 10.28
Output dim: 8, lower bound: -10.7945148, upper bound: 10.7945146
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 10.28
Output dim: 8, lower bound: -10.7041013, upper bound: 10.7041017
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 10.28
Output dim: 8, lower bound: -10.7041020, upper bound: 10.7041011
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 10.28
Output dim: 8, lower bound: -10.7437054, upper bound: 10.7437054
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 10.28
Output dim: 8, lower bound: -10.7437054, upper bound: 10.7437054
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 10.28
Output dim: 8, lower bound: -10.4532328, upper bound: 10.4532344
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 10.28
Output dim: 8, lower bound: -10.4532328, upper bound: 10.4532344
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 10.28
Output dim: 8, lower bound: -10.7437054, upper bound: 10.7437054
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 10.28
Output dim: 8, lower bound: -10.7437054, upper bound: 10.7437054
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.28
Output dim: 8, lower bound: -10.7437054, upper bound: 10.7437054
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.28
Output dim: 8, lower bound: -10.5541862, upper bound: 10.5541874
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.28
Output dim: 8, lower bound: -10.5541862, upper bound: 10.5541874
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.28
Output dim: 8, lower bound: -10.6189809, upper bound: 10.6189809
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.28
Output dim: 8, lower bound: -10.6189809, upper bound: 10.6189809
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.28
Output dim: 8, lower bound: -10.5630472, upper bound: 10.5630465
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.28
Output dim: 8, lower bound: -10.5630465, upper bound: 10.5630466
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.28
Output dim: 8, lower bound: -10.5381283, upper bound: 10.5381283
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.28
Output dim: 8, lower bound: -10.5381283, upper bound: 10.5381283
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.28
Output dim: 8, lower bound: -10.4615045, upper bound: 10.4615045
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.28
Output dim: 8, lower bound: -10.4615045, upper bound: 10.4615045
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.28
Output dim: 8, lower bound: -10.6620022, upper bound: 10.6620023
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.28
Output dim: 8, lower bound: -10.6620022, upper bound: 10.6620023
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.28
Output dim: 8, lower bound: -10.6825183, upper bound: 10.6825183
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.28
Output dim: 8, lower bound: -10.6825183, upper bound: 10.6825183
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.28
Output dim: 8, lower bound: -10.6825183, upper bound: 10.6825183
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.28
Output dim: 8, lower bound: -10.6825183, upper bound: 10.6825183
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.28
Output dim: 8, lower bound: -10.3363539, upper bound: 10.3363539
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.28
Output dim: 8, lower bound: -10.3363539, upper bound: 10.3363539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.28
Output dim: 8, lower bound: -10.8478890, upper bound: 10.8478890
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.28
Output dim: 8, lower bound: -10.8478890, upper bound: 10.8478890
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.28
Output dim: 8, lower bound: -10.8469460, upper bound: 10.8469460
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.28
Output dim: 8, lower bound: -10.8469460, upper bound: 10.8469460
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.28
Output dim: 8, lower bound: -10.8443905, upper bound: 10.8443921
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.28
Output dim: 8, lower bound: -10.8443905, upper bound: 10.8443921
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.28
Output dim: 8, lower bound: -10.6787946, upper bound: 10.6787948
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.28
Output dim: 8, lower bound: -10.6787946, upper bound: 10.6787948
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.28
Output dim: 8, lower bound: -10.5760306, upper bound: 10.5760314
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.28
Output dim: 8, lower bound: -10.5760306, upper bound: 10.5760314
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.28
Output dim: 8, lower bound: -10.6787946, upper bound: 10.6787948
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.28
Output dim: 8, lower bound: -10.6787946, upper bound: 10.6787948
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.28
Output dim: 8, lower bound: -10.4438576, upper bound: 10.4438576
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.28
Output dim: 8, lower bound: -10.4438576, upper bound: 10.4438576

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 6.87 + 597.67 = 604.54 seconds
