## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 10.310653145


## IAR start

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
execution time: IAR + RelationalAnalysis = 0.86 + 5.15 = 6.01 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -10.8533189, upper bound: 10.8533189

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8521238, upper bound: 10.8521232
time: 3.92 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8521232, upper bound: 10.8521243
time: 2.24 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 6.18 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 6.18
Output dim: 8, lower bound: -10.8521238, upper bound: 10.8521232
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 6.18
Output dim: 8, lower bound: -10.8521232, upper bound: 10.8521243

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 218

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8479797, upper bound: 10.8479797
time: 2.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8479797, upper bound: 10.8479797
time: 2.74 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8519560, upper bound: 10.8519564
time: 1.90 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8519559, upper bound: 10.8519566
time: 5.47 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 8.18 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 8.18
Output dim: 8, lower bound: -10.8479797, upper bound: 10.8479797
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 8.18
Output dim: 8, lower bound: -10.8479797, upper bound: 10.8479797
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 8.18
Output dim: 8, lower bound: -10.8519560, upper bound: 10.8519564
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 8.18
Output dim: 8, lower bound: -10.8519559, upper bound: 10.8519566

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 182

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5688875, upper bound: 10.5688874
time: 1.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5688875, upper bound: 10.5688874
time: 1.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6943607, upper bound: 10.6943614
time: 1.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6943607, upper bound: 10.6943614
time: 1.96 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8507852, upper bound: 10.8507845
time: 3.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8507851, upper bound: 10.8507842
time: 3.76 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5879332, upper bound: 10.5879297
time: 1.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5879332, upper bound: 10.5879297
time: 1.31 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 3.51 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.51
Output dim: 8, lower bound: -10.5688875, upper bound: 10.5688874
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.51
Output dim: 8, lower bound: -10.5688875, upper bound: 10.5688874
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.51
Output dim: 8, lower bound: -10.6943607, upper bound: 10.6943614
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.51
Output dim: 8, lower bound: -10.6943607, upper bound: 10.6943614
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.51
Output dim: 8, lower bound: -10.8507852, upper bound: 10.8507845
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.51
Output dim: 8, lower bound: -10.8507851, upper bound: 10.8507842
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.51
Output dim: 8, lower bound: -10.5879332, upper bound: 10.5879297
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.51
Output dim: 8, lower bound: -10.5879332, upper bound: 10.5879297

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5446130, upper bound: 10.5446134
time: 1.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5446130, upper bound: 10.5446134
time: 1.34 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5110101, upper bound: 10.5110101
time: 1.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5110101, upper bound: 10.5110101
time: 1.87 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 126

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 103

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6943607, upper bound: 10.6943614
time: 1.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6943606, upper bound: 10.6943614
time: 2.94 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6373849, upper bound: 10.6373853
time: 1.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6373849, upper bound: 10.6373853
time: 1.50 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8489171, upper bound: 10.8489171
time: 2.13 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8489171, upper bound: 10.8489171
time: 2.05 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 77

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8507851, upper bound: 10.8507842
time: 2.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8507851, upper bound: 10.8507841
time: 4.01 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 218

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3976661, upper bound: 10.3976661
time: 1.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3976661, upper bound: 10.3976661
time: 1.16 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 77

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5879332, upper bound: 10.5879295
time: 1.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5879319, upper bound: 10.5879297
time: 1.59 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 4.14 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.14
Output dim: 8, lower bound: -10.5446130, upper bound: 10.5446134
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.14
Output dim: 8, lower bound: -10.5446130, upper bound: 10.5446134
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.14
Output dim: 8, lower bound: -10.5110101, upper bound: 10.5110101
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.14
Output dim: 8, lower bound: -10.5110101, upper bound: 10.5110101
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.14
Output dim: 8, lower bound: -10.6943607, upper bound: 10.6943614
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.14
Output dim: 8, lower bound: -10.6943606, upper bound: 10.6943614
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.14
Output dim: 8, lower bound: -10.6373849, upper bound: 10.6373853
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.14
Output dim: 8, lower bound: -10.6373849, upper bound: 10.6373853
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.14
Output dim: 8, lower bound: -10.8489171, upper bound: 10.8489171
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.14
Output dim: 8, lower bound: -10.8489171, upper bound: 10.8489171
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.14
Output dim: 8, lower bound: -10.8507851, upper bound: 10.8507842
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.14
Output dim: 8, lower bound: -10.8507851, upper bound: 10.8507841
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.14
Output dim: 8, lower bound: -10.3976661, upper bound: 10.3976661
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.14
Output dim: 8, lower bound: -10.3976661, upper bound: 10.3976661
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.14
Output dim: 8, lower bound: -10.5879332, upper bound: 10.5879295
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.14
Output dim: 8, lower bound: -10.5879319, upper bound: 10.5879297

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5180855, upper bound: 10.5180855
time: 1.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5180855, upper bound: 10.5180855
time: 1.54 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 126

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 123

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3633149, upper bound: 10.3633149
time: 2.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3633149, upper bound: 10.3633149
time: 1.34 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4431143, upper bound: 10.4431143
time: 1.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4431143, upper bound: 10.4431143
time: 1.47 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 92

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5110101, upper bound: 10.5110101
time: 1.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5110101, upper bound: 10.5110101
time: 1.10 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Candidate
type: DSZ, layer: 1, pos: 214

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4081060, upper bound: 10.4081060
time: 1.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4081060, upper bound: 10.4081060
time: 1.62 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6362252, upper bound: 10.6362266
time: 2.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6362259, upper bound: 10.6362259
time: 2.01 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 215

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 214

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 61

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 77

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6373849, upper bound: 10.6373841
time: 1.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6373841, upper bound: 10.6373853
time: 1.82 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4813120, upper bound: 10.4813112
time: 1.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4813112, upper bound: 10.4813127
time: 1.37 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 210

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8255260, upper bound: 10.8255237
time: 1.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8255260, upper bound: 10.8255237
time: 1.67 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8481004, upper bound: 10.8481004
time: 2.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8481003, upper bound: 10.8481004
time: 5.45 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 92

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8507851, upper bound: 10.8507841
time: 2.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8507851, upper bound: 10.8507854
time: 4.98 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3976661, upper bound: 10.3976661
time: 1.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3976661, upper bound: 10.3976661
time: 1.34 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3199734, upper bound: 10.3199735
time: 1.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3199734, upper bound: 10.3199745
time: 1.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 103

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.0298545, upper bound: 10.0298545
time: 1.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.0298545, upper bound: 10.0298545
time: 1.58 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5801414, upper bound: 10.5801414
time: 1.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5801414, upper bound: 10.5801414
time: 1.52 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5515792, upper bound: 10.5515792
time: 1.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5515792, upper bound: 10.5515792
time: 1.56 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 3.93 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.93
Output dim: 8, lower bound: -10.5180855, upper bound: 10.5180855
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.93
Output dim: 8, lower bound: -10.5180855, upper bound: 10.5180855
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.93
Output dim: 8, lower bound: -10.3633149, upper bound: 10.3633149
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.93
Output dim: 8, lower bound: -10.3633149, upper bound: 10.3633149
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.93
Output dim: 8, lower bound: -10.4431143, upper bound: 10.4431143
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.93
Output dim: 8, lower bound: -10.4431143, upper bound: 10.4431143
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.93
Output dim: 8, lower bound: -10.5110101, upper bound: 10.5110101
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.93
Output dim: 8, lower bound: -10.5110101, upper bound: 10.5110101
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.93
Output dim: 8, lower bound: -10.4081060, upper bound: 10.4081060
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.93
Output dim: 8, lower bound: -10.4081060, upper bound: 10.4081060
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.93
Output dim: 8, lower bound: -10.6362252, upper bound: 10.6362266
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.93
Output dim: 8, lower bound: -10.6362259, upper bound: 10.6362259
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.93
Output dim: 8, lower bound: -10.6373849, upper bound: 10.6373841
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.93
Output dim: 8, lower bound: -10.6373841, upper bound: 10.6373853
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.93
Output dim: 8, lower bound: -10.4813120, upper bound: 10.4813112
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.93
Output dim: 8, lower bound: -10.4813112, upper bound: 10.4813127
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.93
Output dim: 8, lower bound: -10.8255260, upper bound: 10.8255237
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.93
Output dim: 8, lower bound: -10.8255260, upper bound: 10.8255237
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.93
Output dim: 8, lower bound: -10.8481004, upper bound: 10.8481004
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.93
Output dim: 8, lower bound: -10.8481003, upper bound: 10.8481004
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.93
Output dim: 8, lower bound: -10.8507851, upper bound: 10.8507841
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.93
Output dim: 8, lower bound: -10.8507851, upper bound: 10.8507854
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.93
Output dim: 8, lower bound: -10.3976661, upper bound: 10.3976661
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.93
Output dim: 8, lower bound: -10.3976661, upper bound: 10.3976661
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.93
Output dim: 8, lower bound: -10.3199734, upper bound: 10.3199735
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.93
Output dim: 8, lower bound: -10.3199734, upper bound: 10.3199745
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.93
Output dim: 8, lower bound: -10.0298545, upper bound: 10.0298545
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.93
Output dim: 8, lower bound: -10.0298545, upper bound: 10.0298545
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.93
Output dim: 8, lower bound: -10.5801414, upper bound: 10.5801414
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.93
Output dim: 8, lower bound: -10.5801414, upper bound: 10.5801414
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.93
Output dim: 8, lower bound: -10.5515792, upper bound: 10.5515792
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.93
Output dim: 8, lower bound: -10.5515792, upper bound: 10.5515792

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4619526, upper bound: 10.4619526
time: 1.36 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4619526, upper bound: 10.4619526
time: 1.27 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 148

### Candidate
type: DSZ, layer: 1, pos: 210

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 214

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 123

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3641069, upper bound: 10.3641072
time: 1.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3641079, upper bound: 10.3641069
time: 1.37 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 92

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3633149, upper bound: 10.3633149
time: 1.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3633149, upper bound: 10.3633149
time: 1.80 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 251

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3401650, upper bound: 10.3401650
time: 2.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3401650, upper bound: 10.3401650
time: 2.38 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4198353, upper bound: 10.4198353
time: 1.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4198353, upper bound: 10.4198353
time: 1.34 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Candidate
type: DSZ, layer: 1, pos: 215

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 103

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.2734649, upper bound: 10.2734655
time: 1.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.2734649, upper bound: 10.2734655
time: 1.31 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3567199, upper bound: 10.3567199
time: 1.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3567199, upper bound: 10.3567199
time: 1.14 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 214

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4431136, upper bound: 10.4431148
time: 1.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4431136, upper bound: 10.4431148
time: 1.27 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1831763, upper bound: 10.1831763
time: 1.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1831763, upper bound: 10.1831763
time: 2.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 103

### Candidate
type: DSZ, layer: 1, pos: 61

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 92

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4081060, upper bound: 10.4081060
time: 1.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4081060, upper bound: 10.4081060
time: 1.23 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4724366, upper bound: 10.4724381
time: 1.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4724375, upper bound: 10.4724366
time: 1.34 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4724366, upper bound: 10.4724380
time: 1.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4724379, upper bound: 10.4724366
time: 1.41 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 126

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4781855, upper bound: 10.4781849
time: 1.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4781855, upper bound: 10.4781849
time: 1.57 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 251

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 214

### Candidate
type: DSZ, layer: 1, pos: 61

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6160070, upper bound: 10.6160076
time: 1.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6160070, upper bound: 10.6160081
time: 1.62 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 126

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 210

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 77

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4813120, upper bound: 10.4813112
time: 2.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4813112, upper bound: 10.4813112
time: 1.36 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.0760721, upper bound: 10.0760733
time: 1.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.0760721, upper bound: 10.0760733
time: 1.73 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7687176, upper bound: 10.7687177
time: 1.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7687176, upper bound: 10.7687177
time: 1.37 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 126

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 77

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8255260, upper bound: 10.8255237
time: 1.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8255260, upper bound: 10.8255237
time: 1.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 103

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7579590, upper bound: 10.7579590
time: 4.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7579590, upper bound: 10.7579590
time: 2.08 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8473394, upper bound: 10.8473395
time: 1.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8473394, upper bound: 10.8473395
time: 1.94 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 61

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8491973, upper bound: 10.8491972
time: 3.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8491973, upper bound: 10.8491972
time: 2.88 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8479725, upper bound: 10.8479715
time: 4.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8479725, upper bound: 10.8479715
time: 3.83 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3869200, upper bound: 10.3869216
time: 1.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3869200, upper bound: 10.3869216
time: 1.90 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 126

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.2965455, upper bound: 10.2965455
time: 1.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.2965455, upper bound: 10.2965455
time: 1.45 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.2860446, upper bound: 10.2860443
time: 1.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.2860446, upper bound: 10.2860443
time: 1.26 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 103

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 211

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.0858939, upper bound: 10.0858939
time: 1.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.0858939, upper bound: 10.0858939
time: 1.13 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 214

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 92

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5801414, upper bound: 10.5801414
time: 2.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5801414, upper bound: 10.5801414
time: 1.36 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5058825, upper bound: 10.5058825
time: 1.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5058825, upper bound: 10.5058825
time: 1.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 126

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4591310, upper bound: 10.4591306
time: 1.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4591310, upper bound: 10.4591306
time: 1.50 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4830883, upper bound: 10.4830905
time: 1.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4830883, upper bound: 10.4830905
time: 1.51 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 3.73 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 8, lower bound: -10.4619526, upper bound: 10.4619526
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 8, lower bound: -10.4619526, upper bound: 10.4619526
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 8, lower bound: -10.3641069, upper bound: 10.3641072
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 8, lower bound: -10.3641079, upper bound: 10.3641069
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 8, lower bound: -10.3633149, upper bound: 10.3633149
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 8, lower bound: -10.3633149, upper bound: 10.3633149
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 8, lower bound: -10.3401650, upper bound: 10.3401650
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 8, lower bound: -10.3401650, upper bound: 10.3401650
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 8, lower bound: -10.4198353, upper bound: 10.4198353
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 8, lower bound: -10.4198353, upper bound: 10.4198353
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.73
Output dim: 8, lower bound: -10.2734649, upper bound: 10.2734655
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.73
Output dim: 8, lower bound: -10.2734649, upper bound: 10.2734655
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 8, lower bound: -10.3567199, upper bound: 10.3567199
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 8, lower bound: -10.3567199, upper bound: 10.3567199
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 8, lower bound: -10.4431136, upper bound: 10.4431148
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 8, lower bound: -10.4431136, upper bound: 10.4431148
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.73
Output dim: 8, lower bound: -10.1831763, upper bound: 10.1831763
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.73
Output dim: 8, lower bound: -10.1831763, upper bound: 10.1831763
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 8, lower bound: -10.4081060, upper bound: 10.4081060
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 8, lower bound: -10.4081060, upper bound: 10.4081060
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 8, lower bound: -10.4724366, upper bound: 10.4724381
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 8, lower bound: -10.4724375, upper bound: 10.4724366
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 8, lower bound: -10.4724366, upper bound: 10.4724380
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 8, lower bound: -10.4724379, upper bound: 10.4724366
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 8, lower bound: -10.4781855, upper bound: 10.4781849
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 8, lower bound: -10.4781855, upper bound: 10.4781849
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 8, lower bound: -10.6160070, upper bound: 10.6160076
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 8, lower bound: -10.6160070, upper bound: 10.6160081
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 8, lower bound: -10.4813120, upper bound: 10.4813112
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 8, lower bound: -10.4813112, upper bound: 10.4813112
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.73
Output dim: 8, lower bound: -10.0760721, upper bound: 10.0760733
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.73
Output dim: 8, lower bound: -10.0760721, upper bound: 10.0760733
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 8, lower bound: -10.7687176, upper bound: 10.7687177
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 8, lower bound: -10.7687176, upper bound: 10.7687177
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 8, lower bound: -10.8255260, upper bound: 10.8255237
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 8, lower bound: -10.8255260, upper bound: 10.8255237
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 8, lower bound: -10.7579590, upper bound: 10.7579590
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 8, lower bound: -10.7579590, upper bound: 10.7579590
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 8, lower bound: -10.8473394, upper bound: 10.8473395
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 8, lower bound: -10.8473394, upper bound: 10.8473395
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 8, lower bound: -10.8491973, upper bound: 10.8491972
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 8, lower bound: -10.8491973, upper bound: 10.8491972
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 8, lower bound: -10.8479725, upper bound: 10.8479715
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 8, lower bound: -10.8479725, upper bound: 10.8479715
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 8, lower bound: -10.3869200, upper bound: 10.3869216
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 8, lower bound: -10.3869200, upper bound: 10.3869216
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.73
Output dim: 8, lower bound: -10.2965455, upper bound: 10.2965455
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.73
Output dim: 8, lower bound: -10.2965455, upper bound: 10.2965455
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.73
Output dim: 8, lower bound: -10.2860446, upper bound: 10.2860443
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.73
Output dim: 8, lower bound: -10.2860446, upper bound: 10.2860443
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.73
Output dim: 8, lower bound: -10.0858939, upper bound: 10.0858939
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.73
Output dim: 8, lower bound: -10.0858939, upper bound: 10.0858939
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 8, lower bound: -10.5801414, upper bound: 10.5801414
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 8, lower bound: -10.5801414, upper bound: 10.5801414
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 8, lower bound: -10.5058825, upper bound: 10.5058825
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 8, lower bound: -10.5058825, upper bound: 10.5058825
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 8, lower bound: -10.4591310, upper bound: 10.4591306
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 8, lower bound: -10.4591310, upper bound: 10.4591306
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 8, lower bound: -10.4830883, upper bound: 10.4830905
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 8, lower bound: -10.4830883, upper bound: 10.4830905

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4374716, upper bound: 10.4374711
time: 1.36 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4374716, upper bound: 10.4374711
time: 1.33 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 103

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 92

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4619526, upper bound: 10.4619526
time: 1.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4619526, upper bound: 10.4619526
time: 1.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 214

### Candidate
type: DSZ, layer: 1, pos: 210

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1807641, upper bound: 10.1807622
time: 1.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1807645, upper bound: 10.1807612
time: 3.10 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 215

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 176

### Candidate
type: DSZ, layer: 1, pos: 167

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1807652, upper bound: 10.1807621
time: 3.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1807655, upper bound: 10.1807612
time: 1.82 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3401651, upper bound: 10.3401650
time: 1.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3401650, upper bound: 10.3401650
time: 1.43 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 210

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 126

### Candidate
type: DSZ, layer: 1, pos: 214

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3401651, upper bound: 10.3401650
time: 1.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3401650, upper bound: 10.3401650
time: 1.34 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1780918, upper bound: 10.1780918
time: 1.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1780918, upper bound: 10.1780918
time: 1.42 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 92

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3401650, upper bound: 10.3401650
time: 1.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3401650, upper bound: 10.3401650
time: 1.26 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 215

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 251

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 214

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 29

### Candidate
type: DSZ, layer: 1, pos: 211

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1560562, upper bound: 10.1560562
time: 1.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1560562, upper bound: 10.1560562
time: 1.46 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 92

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4198353, upper bound: 10.4198353
time: 1.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4198353, upper bound: 10.4198353
time: 1.25 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 214

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.2828764, upper bound: 10.2828764
time: 1.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.2828764, upper bound: 10.2828764
time: 1.21 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3243385, upper bound: 10.3243386
time: 1.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3243385, upper bound: 10.3243386
time: 1.34 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Candidate
type: DSZ, layer: 1, pos: 211

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1794127, upper bound: 10.1794127
time: 1.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1794127, upper bound: 10.1794127
time: 1.39 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 0

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4166158, upper bound: 10.4166151
time: 1.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4166158, upper bound: 10.4166151
time: 1.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1831763, upper bound: 10.1831763
time: 1.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1831763, upper bound: 10.1831763
time: 1.56 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Candidate
type: DSZ, layer: 1, pos: 103

### Candidate
type: DSZ, layer: 1, pos: 214

### Candidate
type: DSZ, layer: 1, pos: 182

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1831763, upper bound: 10.1831763
time: 1.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1831763, upper bound: 10.1831763
time: 2.02 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 126

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.2569304, upper bound: 10.2569318
time: 1.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.2569304, upper bound: 10.2569318
time: 1.65 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 215

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 77

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4724375, upper bound: 10.4724366
time: 1.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4724366, upper bound: 10.4724366
time: 1.78 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4246960, upper bound: 10.4246959
time: 1.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4246960, upper bound: 10.4246959
time: 1.52 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 214

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 211

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1850586, upper bound: 10.1850586
time: 1.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1850586, upper bound: 10.1850591
time: 1.68 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 182

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 215

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.2755800, upper bound: 10.2755800
time: 1.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.2755800, upper bound: 10.2755800
time: 1.45 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.3019601, upper bound: 10.3019613
time: 1.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.3019601, upper bound: 10.3019613
time: 1.35 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4014269, upper bound: 10.4014274
time: 1.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4014269, upper bound: 10.4014274
time: 1.35 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 103

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4014269, upper bound: 10.4014274
time: 1.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4014269, upper bound: 10.4014274
time: 1.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 210

### Candidate
type: DSZ, layer: 1, pos: 126

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.2620956, upper bound: 10.2620956
time: 1.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.2620956, upper bound: 10.2620956
time: 1.25 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3252846, upper bound: 10.3252846
time: 1.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3252846, upper bound: 10.3252846
time: 1.62 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 218

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4417371, upper bound: 10.4417371
time: 1.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4417371, upper bound: 10.4417371
time: 1.25 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 77

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7687176, upper bound: 10.7687177
time: 1.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7687176, upper bound: 10.7687177
time: 1.48 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 92

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8255246, upper bound: 10.8255237
time: 2.02 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8255260, upper bound: 10.8255235
time: 1.87 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 126

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6348870, upper bound: 10.6348870
time: 1.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6348870, upper bound: 10.6348870
time: 1.77 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7128443, upper bound: 10.7128438
time: 1.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7128443, upper bound: 10.7128438
time: 1.54 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 211

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5573288, upper bound: 10.5573259
time: 1.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5573286, upper bound: 10.5573259
time: 1.76 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8448348, upper bound: 10.8448352
time: 3.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8448364, upper bound: 10.8448338
time: 1.42 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 103

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7287271, upper bound: 10.7287266
time: 1.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7287271, upper bound: 10.7287266
time: 1.43 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 251

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7718128, upper bound: 10.7718132
time: 1.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7718128, upper bound: 10.7718132
time: 1.55 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 218

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4194588, upper bound: 10.4194588
time: 1.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4194588, upper bound: 10.4194588
time: 1.24 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8448231, upper bound: 10.8448214
time: 1.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8448231, upper bound: 10.8448214
time: 1.46 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 218

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6194266, upper bound: 10.6194291
time: 1.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6194266, upper bound: 10.6194291
time: 1.59 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 211

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1537051, upper bound: 10.1537051
time: 1.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1537051, upper bound: 10.1537051
time: 1.82 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 103

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3731028, upper bound: 10.3731035
time: 1.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3731028, upper bound: 10.3731035
time: 1.25 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 103

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3931993, upper bound: 10.3931995
time: 1.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3931993, upper bound: 10.3931993
time: 1.65 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 182

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 251

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 215

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 61

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4495379, upper bound: 10.4495363
time: 1.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4495379, upper bound: 10.4495363
time: 1.48 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 9.50 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 9.50
Output dim: 8, lower bound: -10.4374716, upper bound: 10.4374711
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 9.50
Output dim: 8, lower bound: -10.4374716, upper bound: 10.4374711
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 9.50
Output dim: 8, lower bound: -10.4619526, upper bound: 10.4619526
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 9.50
Output dim: 8, lower bound: -10.4619526, upper bound: 10.4619526
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 9.50
Output dim: 8, lower bound: -10.1807641, upper bound: 10.1807622
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 9.50
Output dim: 8, lower bound: -10.1807645, upper bound: 10.1807612
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 9.50
Output dim: 8, lower bound: -10.1807652, upper bound: 10.1807621
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 9.50
Output dim: 8, lower bound: -10.1807655, upper bound: 10.1807612
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 9.50
Output dim: 8, lower bound: -10.3401651, upper bound: 10.3401650
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 9.50
Output dim: 8, lower bound: -10.3401650, upper bound: 10.3401650
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 9.50
Output dim: 8, lower bound: -10.3401651, upper bound: 10.3401650
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 9.50
Output dim: 8, lower bound: -10.3401650, upper bound: 10.3401650
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 9.50
Output dim: 8, lower bound: -10.1780918, upper bound: 10.1780918
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 9.50
Output dim: 8, lower bound: -10.1780918, upper bound: 10.1780918
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 9.50
Output dim: 8, lower bound: -10.3401650, upper bound: 10.3401650
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 9.50
Output dim: 8, lower bound: -10.3401650, upper bound: 10.3401650
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 9.50
Output dim: 8, lower bound: -10.1560562, upper bound: 10.1560562
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 9.50
Output dim: 8, lower bound: -10.1560562, upper bound: 10.1560562
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 9.50
Output dim: 8, lower bound: -10.4198353, upper bound: 10.4198353
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 9.50
Output dim: 8, lower bound: -10.4198353, upper bound: 10.4198353
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 9.50
Output dim: 8, lower bound: -10.2828764, upper bound: 10.2828764
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 9.50
Output dim: 8, lower bound: -10.2828764, upper bound: 10.2828764
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 9.50
Output dim: 8, lower bound: -10.3243385, upper bound: 10.3243386
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 9.50
Output dim: 8, lower bound: -10.3243385, upper bound: 10.3243386
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 9.50
Output dim: 8, lower bound: -10.1794127, upper bound: 10.1794127
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 9.50
Output dim: 8, lower bound: -10.1794127, upper bound: 10.1794127
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 9.50
Output dim: 8, lower bound: -10.4166158, upper bound: 10.4166151
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 9.50
Output dim: 8, lower bound: -10.4166158, upper bound: 10.4166151
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 9.50
Output dim: 8, lower bound: -10.1831763, upper bound: 10.1831763
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 9.50
Output dim: 8, lower bound: -10.1831763, upper bound: 10.1831763
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 9.50
Output dim: 8, lower bound: -10.1831763, upper bound: 10.1831763
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 9.50
Output dim: 8, lower bound: -10.1831763, upper bound: 10.1831763
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 9.50
Output dim: 8, lower bound: -10.2569304, upper bound: 10.2569318
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 9.50
Output dim: 8, lower bound: -10.2569304, upper bound: 10.2569318
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 9.50
Output dim: 8, lower bound: -10.4724375, upper bound: 10.4724366
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 9.50
Output dim: 8, lower bound: -10.4724366, upper bound: 10.4724366
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 9.50
Output dim: 8, lower bound: -10.4246960, upper bound: 10.4246959
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 9.50
Output dim: 8, lower bound: -10.4246960, upper bound: 10.4246959
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 9.50
Output dim: 8, lower bound: -10.1850586, upper bound: 10.1850586
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 9.50
Output dim: 8, lower bound: -10.1850586, upper bound: 10.1850591
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 9.50
Output dim: 8, lower bound: -10.2755800, upper bound: 10.2755800
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 9.50
Output dim: 8, lower bound: -10.2755800, upper bound: 10.2755800
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 9.50
Output dim: 8, lower bound: -10.3019601, upper bound: 10.3019613
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 9.50
Output dim: 8, lower bound: -10.3019601, upper bound: 10.3019613
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 9.50
Output dim: 8, lower bound: -10.4014269, upper bound: 10.4014274
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 9.50
Output dim: 8, lower bound: -10.4014269, upper bound: 10.4014274
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 9.50
Output dim: 8, lower bound: -10.4014269, upper bound: 10.4014274
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 9.50
Output dim: 8, lower bound: -10.4014269, upper bound: 10.4014274
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 9.50
Output dim: 8, lower bound: -10.2620956, upper bound: 10.2620956
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 9.50
Output dim: 8, lower bound: -10.2620956, upper bound: 10.2620956
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 9.50
Output dim: 8, lower bound: -10.3252846, upper bound: 10.3252846
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 9.50
Output dim: 8, lower bound: -10.3252846, upper bound: 10.3252846
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 9.50
Output dim: 8, lower bound: -10.4417371, upper bound: 10.4417371
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 9.50
Output dim: 8, lower bound: -10.4417371, upper bound: 10.4417371
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 9.50
Output dim: 8, lower bound: -10.7687176, upper bound: 10.7687177
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 9.50
Output dim: 8, lower bound: -10.7687176, upper bound: 10.7687177
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 9.50
Output dim: 8, lower bound: -10.8255246, upper bound: 10.8255237
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 9.50
Output dim: 8, lower bound: -10.8255260, upper bound: 10.8255235
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 9.50
Output dim: 8, lower bound: -10.6348870, upper bound: 10.6348870
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 9.50
Output dim: 8, lower bound: -10.6348870, upper bound: 10.6348870
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 9.50
Output dim: 8, lower bound: -10.7128443, upper bound: 10.7128438
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 9.50
Output dim: 8, lower bound: -10.7128443, upper bound: 10.7128438
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 9.50
Output dim: 8, lower bound: -10.5573288, upper bound: 10.5573259
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 9.50
Output dim: 8, lower bound: -10.5573286, upper bound: 10.5573259
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 9.50
Output dim: 8, lower bound: -10.8448348, upper bound: 10.8448352
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 9.50
Output dim: 8, lower bound: -10.8448364, upper bound: 10.8448338
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 9.50
Output dim: 8, lower bound: -10.7287271, upper bound: 10.7287266
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 9.50
Output dim: 8, lower bound: -10.7287271, upper bound: 10.7287266
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 9.50
Output dim: 8, lower bound: -10.7718128, upper bound: 10.7718132
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 9.50
Output dim: 8, lower bound: -10.7718128, upper bound: 10.7718132
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 9.50
Output dim: 8, lower bound: -10.4194588, upper bound: 10.4194588
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 9.50
Output dim: 8, lower bound: -10.4194588, upper bound: 10.4194588
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 9.50
Output dim: 8, lower bound: -10.8448231, upper bound: 10.8448214
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 9.50
Output dim: 8, lower bound: -10.8448231, upper bound: 10.8448214
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 9.50
Output dim: 8, lower bound: -10.6194266, upper bound: 10.6194291
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 9.50
Output dim: 8, lower bound: -10.6194266, upper bound: 10.6194291
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 9.50
Output dim: 8, lower bound: -10.1537051, upper bound: 10.1537051
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 9.50
Output dim: 8, lower bound: -10.1537051, upper bound: 10.1537051
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 9.50
Output dim: 8, lower bound: -10.3731028, upper bound: 10.3731035
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 9.50
Output dim: 8, lower bound: -10.3731028, upper bound: 10.3731035
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 9.50
Output dim: 8, lower bound: -10.3931993, upper bound: 10.3931995
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 9.50
Output dim: 8, lower bound: -10.3931993, upper bound: 10.3931993
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 9.50
Output dim: 8, lower bound: -10.4495379, upper bound: 10.4495363
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 9.50
Output dim: 8, lower bound: -10.4495379, upper bound: 10.4495363
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 9.50
Output dim: 8, lower bound: -10.5058825, upper bound: 10.5058825
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 9.50
Output dim: 8, lower bound: -10.5058825, upper bound: 10.5058825
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 9.50
Output dim: 8, lower bound: -10.4591310, upper bound: 10.4591306
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 9.50
Output dim: 8, lower bound: -10.4591310, upper bound: 10.4591306
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 9.50
Output dim: 8, lower bound: -10.4830883, upper bound: 10.4830905
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 9.50
Output dim: 8, lower bound: -10.4830883, upper bound: 10.4830905

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 6.01 + 595.54 = 601.55 seconds
