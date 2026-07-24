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
execution time: IAR + RelationalAnalysis = 0.81 + 5.28 = 6.09 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -10.8533189, upper bound: 10.8533189

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 126

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8476955, upper bound: 10.8476955
time: 1.48 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8476955, upper bound: 10.8476955
time: 1.47 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 3.01 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 3.01
Output dim: 8, lower bound: -10.8476955, upper bound: 10.8476955
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 3.01
Output dim: 8, lower bound: -10.8476955, upper bound: 10.8476955

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 126

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8406201, upper bound: 10.8406201
time: 1.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8406201, upper bound: 10.8406201
time: 1.39 seconds

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 126

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8406201, upper bound: 10.8406201
time: 1.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8406201, upper bound: 10.8406201
time: 1.38 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 3.60 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 3.60
Output dim: 8, lower bound: -10.8406201, upper bound: 10.8406201
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 3.60
Output dim: 8, lower bound: -10.8406201, upper bound: 10.8406201
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 3.60
Output dim: 8, lower bound: -10.8406201, upper bound: 10.8406201
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 3.60
Output dim: 8, lower bound: -10.8406201, upper bound: 10.8406201

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 126

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 215

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6182344, upper bound: 10.6182340
time: 1.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6182344, upper bound: 10.6182340
time: 1.36 seconds

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 126

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 215

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6182340, upper bound: 10.6182344
time: 1.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6182340, upper bound: 10.6182344
time: 6.10 seconds

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 126

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 215

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6182344, upper bound: 10.6182340
time: 1.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6182344, upper bound: 10.6182340
time: 1.40 seconds

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 126

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 215

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6182340, upper bound: 10.6182344
time: 1.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6182340, upper bound: 10.6182344
time: 1.86 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 10.07 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 10.07
Output dim: 8, lower bound: -10.6182344, upper bound: 10.6182340
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 10.07
Output dim: 8, lower bound: -10.6182344, upper bound: 10.6182340
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 10.07
Output dim: 8, lower bound: -10.6182340, upper bound: 10.6182344
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 10.07
Output dim: 8, lower bound: -10.6182340, upper bound: 10.6182344
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 10.07
Output dim: 8, lower bound: -10.6182344, upper bound: 10.6182340
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 10.07
Output dim: 8, lower bound: -10.6182344, upper bound: 10.6182340
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 10.07
Output dim: 8, lower bound: -10.6182340, upper bound: 10.6182344
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 10.07
Output dim: 8, lower bound: -10.6182340, upper bound: 10.6182344

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 126

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Candidate
type: DSZ, layer: 1, pos: 215

### Candidate
type: DSZ, layer: 1, pos: 0

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4289461, upper bound: 10.4289461
time: 1.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4289461, upper bound: 10.4289461
time: 1.08 seconds

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 126

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Candidate
type: DSZ, layer: 1, pos: 215

### Candidate
type: DSZ, layer: 1, pos: 0

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4289461, upper bound: 10.4289461
time: 1.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4289461, upper bound: 10.4289461
time: 1.06 seconds

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 126

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Candidate
type: DSZ, layer: 1, pos: 215

### Candidate
type: DSZ, layer: 1, pos: 0

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4289461, upper bound: 10.4289461
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4289461, upper bound: 10.4289461
time: 1.96 seconds

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 126

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Candidate
type: DSZ, layer: 1, pos: 215

### Candidate
type: DSZ, layer: 1, pos: 0

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4289461, upper bound: 10.4289461
time: 1.05 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4289461, upper bound: 10.4289461
time: 1.91 seconds

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 126

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Candidate
type: DSZ, layer: 1, pos: 215

### Candidate
type: DSZ, layer: 1, pos: 0

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4289461, upper bound: 10.4289461
time: 1.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4289461, upper bound: 10.4289461
time: 1.07 seconds

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 126

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Candidate
type: DSZ, layer: 1, pos: 215

### Candidate
type: DSZ, layer: 1, pos: 0

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4289461, upper bound: 10.4289461
time: 1.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4289461, upper bound: 10.4289461
time: 1.07 seconds

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 126

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Candidate
type: DSZ, layer: 1, pos: 215

### Candidate
type: DSZ, layer: 1, pos: 0

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4289461, upper bound: 10.4289461
time: 1.05 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4289461, upper bound: 10.4289461
time: 1.92 seconds

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 126

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Candidate
type: DSZ, layer: 1, pos: 215

### Candidate
type: DSZ, layer: 1, pos: 0

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4289461, upper bound: 10.4289461
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4289461, upper bound: 10.4289461
time: 1.91 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 5.62 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.62
Output dim: 8, lower bound: -10.4289461, upper bound: 10.4289461
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.62
Output dim: 8, lower bound: -10.4289461, upper bound: 10.4289461
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.62
Output dim: 8, lower bound: -10.4289461, upper bound: 10.4289461
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.62
Output dim: 8, lower bound: -10.4289461, upper bound: 10.4289461
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.62
Output dim: 8, lower bound: -10.4289461, upper bound: 10.4289461
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.62
Output dim: 8, lower bound: -10.4289461, upper bound: 10.4289461
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.62
Output dim: 8, lower bound: -10.4289461, upper bound: 10.4289461
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.62
Output dim: 8, lower bound: -10.4289461, upper bound: 10.4289461
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.62
Output dim: 8, lower bound: -10.4289461, upper bound: 10.4289461
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.62
Output dim: 8, lower bound: -10.4289461, upper bound: 10.4289461
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.62
Output dim: 8, lower bound: -10.4289461, upper bound: 10.4289461
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.62
Output dim: 8, lower bound: -10.4289461, upper bound: 10.4289461
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.62
Output dim: 8, lower bound: -10.4289461, upper bound: 10.4289461
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.62
Output dim: 8, lower bound: -10.4289461, upper bound: 10.4289461
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.62
Output dim: 8, lower bound: -10.4289461, upper bound: 10.4289461
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.62
Output dim: 8, lower bound: -10.4289461, upper bound: 10.4289461

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 126

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Candidate
type: DSZ, layer: 1, pos: 215

### Candidate
type: DSZ, layer: 1, pos: 0

### Candidate
type: DSZ, layer: 1, pos: 144

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3451639, upper bound: 10.3451647
time: 1.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3451639, upper bound: 10.3451647
time: 1.33 seconds

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 126

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Candidate
type: DSZ, layer: 1, pos: 215

### Candidate
type: DSZ, layer: 1, pos: 0

### Candidate
type: DSZ, layer: 1, pos: 144

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3451639, upper bound: 10.3451639
time: 1.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3451639, upper bound: 10.3451639
time: 1.31 seconds

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 126

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Candidate
type: DSZ, layer: 1, pos: 215

### Candidate
type: DSZ, layer: 1, pos: 0

### Candidate
type: DSZ, layer: 1, pos: 144

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3451639, upper bound: 10.3451647
time: 1.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3451639, upper bound: 10.3451647
time: 1.33 seconds

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 126

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Candidate
type: DSZ, layer: 1, pos: 215

### Candidate
type: DSZ, layer: 1, pos: 0

### Candidate
type: DSZ, layer: 1, pos: 144

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3451639, upper bound: 10.3451639
time: 1.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3451639, upper bound: 10.3451639
time: 1.32 seconds

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 126

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Candidate
type: DSZ, layer: 1, pos: 215

### Candidate
type: DSZ, layer: 1, pos: 0

### Candidate
type: DSZ, layer: 1, pos: 144

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3451639, upper bound: 10.3451639
time: 1.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3451639, upper bound: 10.3451639
time: 1.99 seconds

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 126

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Candidate
type: DSZ, layer: 1, pos: 215

### Candidate
type: DSZ, layer: 1, pos: 0

### Candidate
type: DSZ, layer: 1, pos: 144

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3451647, upper bound: 10.3451639
time: 1.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3451647, upper bound: 10.3451639
time: 1.33 seconds

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 126

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Candidate
type: DSZ, layer: 1, pos: 215

### Candidate
type: DSZ, layer: 1, pos: 0

### Candidate
type: DSZ, layer: 1, pos: 144

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3451639, upper bound: 10.3451639
time: 1.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3451639, upper bound: 10.3451639
time: 2.27 seconds

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 126

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Candidate
type: DSZ, layer: 1, pos: 215

### Candidate
type: DSZ, layer: 1, pos: 0

### Candidate
type: DSZ, layer: 1, pos: 144

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3451647, upper bound: 10.3451639
time: 1.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3451647, upper bound: 10.3451639
time: 2.64 seconds

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 126

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Candidate
type: DSZ, layer: 1, pos: 215

### Candidate
type: DSZ, layer: 1, pos: 0

### Candidate
type: DSZ, layer: 1, pos: 144

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3451639, upper bound: 10.3451647
time: 1.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3451639, upper bound: 10.3451647
time: 1.41 seconds

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 126

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Candidate
type: DSZ, layer: 1, pos: 215

### Candidate
type: DSZ, layer: 1, pos: 0

### Candidate
type: DSZ, layer: 1, pos: 144

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3451639, upper bound: 10.3451639
time: 1.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3451639, upper bound: 10.3451639
time: 1.29 seconds

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 126

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Candidate
type: DSZ, layer: 1, pos: 215

### Candidate
type: DSZ, layer: 1, pos: 0

### Candidate
type: DSZ, layer: 1, pos: 144

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3451639, upper bound: 10.3451647
time: 1.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3451639, upper bound: 10.3451647
time: 1.36 seconds

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 126

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Candidate
type: DSZ, layer: 1, pos: 215

### Candidate
type: DSZ, layer: 1, pos: 0

### Candidate
type: DSZ, layer: 1, pos: 144

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3451639, upper bound: 10.3451639
time: 1.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3451639, upper bound: 10.3451639
time: 1.29 seconds

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 126

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Candidate
type: DSZ, layer: 1, pos: 215

### Candidate
type: DSZ, layer: 1, pos: 0

### Candidate
type: DSZ, layer: 1, pos: 144

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3451639, upper bound: 10.3451639
time: 1.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3451639, upper bound: 10.3451639
time: 1.62 seconds

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 126

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Candidate
type: DSZ, layer: 1, pos: 215

### Candidate
type: DSZ, layer: 1, pos: 0

### Candidate
type: DSZ, layer: 1, pos: 144

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3451647, upper bound: 10.3451639
time: 1.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3451647, upper bound: 10.3451639
time: 1.32 seconds

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 126

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Candidate
type: DSZ, layer: 1, pos: 215

### Candidate
type: DSZ, layer: 1, pos: 0

### Candidate
type: DSZ, layer: 1, pos: 144

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3451639, upper bound: 10.3451639
time: 1.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3451639, upper bound: 10.3451639
time: 1.61 seconds

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 126

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Candidate
type: DSZ, layer: 1, pos: 215

### Candidate
type: DSZ, layer: 1, pos: 0

### Candidate
type: DSZ, layer: 1, pos: 144

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3451647, upper bound: 10.3451639
time: 1.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3451647, upper bound: 10.3451639
time: 3.04 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 7.22 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 7.22
Output dim: 8, lower bound: -10.3451639, upper bound: 10.3451647
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 7.22
Output dim: 8, lower bound: -10.3451639, upper bound: 10.3451647
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 7.22
Output dim: 8, lower bound: -10.3451639, upper bound: 10.3451639
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 7.22
Output dim: 8, lower bound: -10.3451639, upper bound: 10.3451639
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 7.22
Output dim: 8, lower bound: -10.3451639, upper bound: 10.3451647
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 7.22
Output dim: 8, lower bound: -10.3451639, upper bound: 10.3451647
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 7.22
Output dim: 8, lower bound: -10.3451639, upper bound: 10.3451639
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 7.22
Output dim: 8, lower bound: -10.3451639, upper bound: 10.3451639
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 7.22
Output dim: 8, lower bound: -10.3451639, upper bound: 10.3451639
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 7.22
Output dim: 8, lower bound: -10.3451639, upper bound: 10.3451639
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 7.22
Output dim: 8, lower bound: -10.3451647, upper bound: 10.3451639
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 7.22
Output dim: 8, lower bound: -10.3451647, upper bound: 10.3451639
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 7.22
Output dim: 8, lower bound: -10.3451639, upper bound: 10.3451639
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 7.22
Output dim: 8, lower bound: -10.3451639, upper bound: 10.3451639
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 7.22
Output dim: 8, lower bound: -10.3451647, upper bound: 10.3451639
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 7.22
Output dim: 8, lower bound: -10.3451647, upper bound: 10.3451639
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 7.22
Output dim: 8, lower bound: -10.3451639, upper bound: 10.3451647
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 7.22
Output dim: 8, lower bound: -10.3451639, upper bound: 10.3451647
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 7.22
Output dim: 8, lower bound: -10.3451639, upper bound: 10.3451639
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 7.22
Output dim: 8, lower bound: -10.3451639, upper bound: 10.3451639
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 7.22
Output dim: 8, lower bound: -10.3451639, upper bound: 10.3451647
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 7.22
Output dim: 8, lower bound: -10.3451639, upper bound: 10.3451647
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 7.22
Output dim: 8, lower bound: -10.3451639, upper bound: 10.3451639
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 7.22
Output dim: 8, lower bound: -10.3451639, upper bound: 10.3451639
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 7.22
Output dim: 8, lower bound: -10.3451639, upper bound: 10.3451639
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 7.22
Output dim: 8, lower bound: -10.3451639, upper bound: 10.3451639
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 7.22
Output dim: 8, lower bound: -10.3451647, upper bound: 10.3451639
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 7.22
Output dim: 8, lower bound: -10.3451647, upper bound: 10.3451639
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 7.22
Output dim: 8, lower bound: -10.3451639, upper bound: 10.3451639
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 7.22
Output dim: 8, lower bound: -10.3451639, upper bound: 10.3451639
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 7.22
Output dim: 8, lower bound: -10.3451647, upper bound: 10.3451639
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 7.22
Output dim: 8, lower bound: -10.3451647, upper bound: 10.3451639

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
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 126

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Candidate
type: DSZ, layer: 1, pos: 215

### Candidate
type: DSZ, layer: 1, pos: 0

### Candidate
type: DSZ, layer: 1, pos: 144

### Candidate
type: DSZ, layer: 1, pos: 176

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
time: 1.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
time: 1.86 seconds

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 126

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Candidate
type: DSZ, layer: 1, pos: 215

### Candidate
type: DSZ, layer: 1, pos: 0

### Candidate
type: DSZ, layer: 1, pos: 144

### Candidate
type: DSZ, layer: 1, pos: 176

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
time: 1.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
time: 1.74 seconds

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 126

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Candidate
type: DSZ, layer: 1, pos: 215

### Candidate
type: DSZ, layer: 1, pos: 0

### Candidate
type: DSZ, layer: 1, pos: 144

### Candidate
type: DSZ, layer: 1, pos: 176

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
time: 1.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
time: 1.74 seconds

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
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 126

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Candidate
type: DSZ, layer: 1, pos: 215

### Candidate
type: DSZ, layer: 1, pos: 0

### Candidate
type: DSZ, layer: 1, pos: 144

### Candidate
type: DSZ, layer: 1, pos: 176

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
time: 1.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
time: 1.61 seconds

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 126

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Candidate
type: DSZ, layer: 1, pos: 215

### Candidate
type: DSZ, layer: 1, pos: 0

### Candidate
type: DSZ, layer: 1, pos: 144

### Candidate
type: DSZ, layer: 1, pos: 176

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
time: 1.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
time: 1.55 seconds

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 126

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Candidate
type: DSZ, layer: 1, pos: 215

### Candidate
type: DSZ, layer: 1, pos: 0

### Candidate
type: DSZ, layer: 1, pos: 144

### Candidate
type: DSZ, layer: 1, pos: 176

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
time: 1.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
time: 1.86 seconds

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 126

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Candidate
type: DSZ, layer: 1, pos: 215

### Candidate
type: DSZ, layer: 1, pos: 0

### Candidate
type: DSZ, layer: 1, pos: 144

### Candidate
type: DSZ, layer: 1, pos: 176

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
time: 1.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
time: 2.20 seconds

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 126

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Candidate
type: DSZ, layer: 1, pos: 215

### Candidate
type: DSZ, layer: 1, pos: 0

### Candidate
type: DSZ, layer: 1, pos: 144

### Candidate
type: DSZ, layer: 1, pos: 176

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
time: 1.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
time: 1.59 seconds

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 126

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Candidate
type: DSZ, layer: 1, pos: 215

### Candidate
type: DSZ, layer: 1, pos: 0

### Candidate
type: DSZ, layer: 1, pos: 144

### Candidate
type: DSZ, layer: 1, pos: 176

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
time: 2.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
time: 1.60 seconds

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 126

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Candidate
type: DSZ, layer: 1, pos: 215

### Candidate
type: DSZ, layer: 1, pos: 0

### Candidate
type: DSZ, layer: 1, pos: 144

### Candidate
type: DSZ, layer: 1, pos: 176

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
time: 1.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
time: 1.68 seconds

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 126

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Candidate
type: DSZ, layer: 1, pos: 215

### Candidate
type: DSZ, layer: 1, pos: 0

### Candidate
type: DSZ, layer: 1, pos: 144

### Candidate
type: DSZ, layer: 1, pos: 176

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
time: 1.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
time: 1.57 seconds

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 126

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Candidate
type: DSZ, layer: 1, pos: 215

### Candidate
type: DSZ, layer: 1, pos: 0

### Candidate
type: DSZ, layer: 1, pos: 144

### Candidate
type: DSZ, layer: 1, pos: 176

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
time: 1.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
time: 1.87 seconds

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
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 126

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Candidate
type: DSZ, layer: 1, pos: 215

### Candidate
type: DSZ, layer: 1, pos: 0

### Candidate
type: DSZ, layer: 1, pos: 144

### Candidate
type: DSZ, layer: 1, pos: 176

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
time: 2.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
time: 1.77 seconds

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 126

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Candidate
type: DSZ, layer: 1, pos: 215

### Candidate
type: DSZ, layer: 1, pos: 0

### Candidate
type: DSZ, layer: 1, pos: 144

### Candidate
type: DSZ, layer: 1, pos: 176

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
time: 1.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
time: 1.90 seconds

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 126

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Candidate
type: DSZ, layer: 1, pos: 215

### Candidate
type: DSZ, layer: 1, pos: 0

### Candidate
type: DSZ, layer: 1, pos: 144

### Candidate
type: DSZ, layer: 1, pos: 176

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
time: 1.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
time: 1.50 seconds

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 126

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Candidate
type: DSZ, layer: 1, pos: 215

### Candidate
type: DSZ, layer: 1, pos: 0

### Candidate
type: DSZ, layer: 1, pos: 144

### Candidate
type: DSZ, layer: 1, pos: 176

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
time: 1.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 126

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Candidate
type: DSZ, layer: 1, pos: 215

### Candidate
type: DSZ, layer: 1, pos: 0

### Candidate
type: DSZ, layer: 1, pos: 144

### Candidate
type: DSZ, layer: 1, pos: 176

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
time: 1.34 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
time: 1.82 seconds

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 126

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Candidate
type: DSZ, layer: 1, pos: 215

### Candidate
type: DSZ, layer: 1, pos: 0

### Candidate
type: DSZ, layer: 1, pos: 144

### Candidate
type: DSZ, layer: 1, pos: 176

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
time: 1.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
time: 2.08 seconds

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

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 126

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Candidate
type: DSZ, layer: 1, pos: 215

### Candidate
type: DSZ, layer: 1, pos: 0

### Candidate
type: DSZ, layer: 1, pos: 144

### Candidate
type: DSZ, layer: 1, pos: 176

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
time: 1.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
time: 1.77 seconds

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

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 126

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Candidate
type: DSZ, layer: 1, pos: 215

### Candidate
type: DSZ, layer: 1, pos: 0

### Candidate
type: DSZ, layer: 1, pos: 144

### Candidate
type: DSZ, layer: 1, pos: 176

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
time: 1.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
time: 1.73 seconds

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

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 126

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Candidate
type: DSZ, layer: 1, pos: 215

### Candidate
type: DSZ, layer: 1, pos: 0

### Candidate
type: DSZ, layer: 1, pos: 144

### Candidate
type: DSZ, layer: 1, pos: 176

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
time: 1.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
time: 1.63 seconds

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 126

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Candidate
type: DSZ, layer: 1, pos: 215

### Candidate
type: DSZ, layer: 1, pos: 0

### Candidate
type: DSZ, layer: 1, pos: 144

### Candidate
type: DSZ, layer: 1, pos: 176

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
time: 1.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
time: 1.93 seconds

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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 126

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Candidate
type: DSZ, layer: 1, pos: 215

### Candidate
type: DSZ, layer: 1, pos: 0

### Candidate
type: DSZ, layer: 1, pos: 144

### Candidate
type: DSZ, layer: 1, pos: 176

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
time: 1.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
time: 2.24 seconds

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

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 126

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Candidate
type: DSZ, layer: 1, pos: 215

### Candidate
type: DSZ, layer: 1, pos: 0

### Candidate
type: DSZ, layer: 1, pos: 144

### Candidate
type: DSZ, layer: 1, pos: 176

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
time: 1.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
time: 1.79 seconds

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

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 126

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Candidate
type: DSZ, layer: 1, pos: 215

### Candidate
type: DSZ, layer: 1, pos: 0

### Candidate
type: DSZ, layer: 1, pos: 144

### Candidate
type: DSZ, layer: 1, pos: 176

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
time: 2.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
time: 1.66 seconds

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 126

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Candidate
type: DSZ, layer: 1, pos: 215

### Candidate
type: DSZ, layer: 1, pos: 0

### Candidate
type: DSZ, layer: 1, pos: 144

### Candidate
type: DSZ, layer: 1, pos: 176

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
time: 1.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
time: 1.93 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

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
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 126

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Candidate
type: DSZ, layer: 1, pos: 215

### Candidate
type: DSZ, layer: 1, pos: 0

### Candidate
type: DSZ, layer: 1, pos: 144

### Candidate
type: DSZ, layer: 1, pos: 176

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
time: 1.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
time: 1.54 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 126

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Candidate
type: DSZ, layer: 1, pos: 215

### Candidate
type: DSZ, layer: 1, pos: 0

### Candidate
type: DSZ, layer: 1, pos: 144

### Candidate
type: DSZ, layer: 1, pos: 176

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
time: 1.34 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
time: 3.84 seconds

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

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 126

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Candidate
type: DSZ, layer: 1, pos: 215

### Candidate
type: DSZ, layer: 1, pos: 0

### Candidate
type: DSZ, layer: 1, pos: 144

### Candidate
type: DSZ, layer: 1, pos: 176

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
time: 2.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
time: 1.89 seconds

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

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 126

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Candidate
type: DSZ, layer: 1, pos: 215

### Candidate
type: DSZ, layer: 1, pos: 0

### Candidate
type: DSZ, layer: 1, pos: 144

### Candidate
type: DSZ, layer: 1, pos: 176

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
time: 1.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
time: 1.95 seconds

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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 126

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Candidate
type: DSZ, layer: 1, pos: 215

### Candidate
type: DSZ, layer: 1, pos: 0

### Candidate
type: DSZ, layer: 1, pos: 144

### Candidate
type: DSZ, layer: 1, pos: 176

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
time: 1.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
time: 1.62 seconds

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

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 126

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Candidate
type: DSZ, layer: 1, pos: 215

### Candidate
type: DSZ, layer: 1, pos: 0

### Candidate
type: DSZ, layer: 1, pos: 144

### Candidate
type: DSZ, layer: 1, pos: 176

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
time: 1.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
time: 4.07 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 6.37 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.1720631, upper bound: 10.1720631

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 6.09 + 325.36 = 331.45 seconds
