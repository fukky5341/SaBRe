## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.001357455


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625)
1: (-0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128)
2: (0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554)
3: (-0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0126610, 0.0126610)
4: (-0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531)
5: (0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450)
6: (0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107)
7: (-0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458)
8: (0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171)
9: (0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0053181, 0.0053181)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.08 + 2.41 = 3.49 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -0.0014289, upper bound: 0.0014289

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014267, upper bound: 0.0014270
time: 1.61 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014269, upper bound: 0.0014268
time: 1.16 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 2.86 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 2.86
Output dim: 2, lower bound: -0.0014267, upper bound: 0.0014270
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 2.86
Output dim: 2, lower bound: -0.0014269, upper bound: 0.0014268

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0126144, 0.0126187
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0053138, 0.0053133

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014194, upper bound: 0.0014127
time: 1.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014119, upper bound: 0.0014195
time: 1.56 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0126187, 0.0126144
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0053133, 0.0053138

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014196, upper bound: 0.0014119
time: 1.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014127, upper bound: 0.0014194
time: 1.68 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 4.24 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 4.24
Output dim: 2, lower bound: -0.0014194, upper bound: 0.0014127
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 4.24
Output dim: 2, lower bound: -0.0014119, upper bound: 0.0014195
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 4.24
Output dim: 2, lower bound: -0.0014196, upper bound: 0.0014119
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 4.24
Output dim: 2, lower bound: -0.0014127, upper bound: 0.0014194

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0122754, 0.0121265
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0052571, 0.0052734

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014053, upper bound: 0.0014085
time: 1.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014151, upper bound: 0.0013972
time: 1.25 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0121223, 0.0122776
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0052737, 0.0052566

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013971, upper bound: 0.0014152
time: 1.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014077, upper bound: 0.0014053
time: 1.50 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0122776, 0.0121223
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0052566, 0.0052737

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014053, upper bound: 0.0014077
time: 1.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014152, upper bound: 0.0013971
time: 1.60 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0121265, 0.0122754
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0052734, 0.0052571

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013971, upper bound: 0.0014150
time: 1.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014085, upper bound: 0.0014054
time: 1.50 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 4.21 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 4.21
Output dim: 2, lower bound: -0.0014053, upper bound: 0.0014085
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 4.21
Output dim: 2, lower bound: -0.0014151, upper bound: 0.0013972
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 4.21
Output dim: 2, lower bound: -0.0013971, upper bound: 0.0014152
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 4.21
Output dim: 2, lower bound: -0.0014077, upper bound: 0.0014053
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 4.21
Output dim: 2, lower bound: -0.0014053, upper bound: 0.0014077
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 4.21
Output dim: 2, lower bound: -0.0014152, upper bound: 0.0013971
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 4.21
Output dim: 2, lower bound: -0.0013971, upper bound: 0.0014150
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 4.21
Output dim: 2, lower bound: -0.0014085, upper bound: 0.0014054

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0119818, 0.0119133
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0052347, 0.0052423

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 172

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013855, upper bound: 0.0014056
time: 1.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014023, upper bound: 0.0013837
time: 1.78 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0120476, 0.0118329
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0052259, 0.0052495

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 172

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013908, upper bound: 0.0013941
time: 1.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014121, upper bound: 0.0013770
time: 1.55 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0118287, 0.0120517
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0052499, 0.0052254

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 172

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013769, upper bound: 0.0014123
time: 1.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013940, upper bound: 0.0013908
time: 1.62 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0119072, 0.0119840
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0052425, 0.0052341

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 172

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013835, upper bound: 0.0014022
time: 1.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014047, upper bound: 0.0013854
time: 1.52 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0119840, 0.0119072
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0052341, 0.0052425

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 172

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013854, upper bound: 0.0014048
time: 1.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014023, upper bound: 0.0013835
time: 1.74 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0120517, 0.0118286
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0052254, 0.0052499

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 172

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013908, upper bound: 0.0013940
time: 1.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014123, upper bound: 0.0013770
time: 1.54 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0118329, 0.0120476
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0052495, 0.0052259

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 172

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013770, upper bound: 0.0014121
time: 1.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013941, upper bound: 0.0013908
time: 1.47 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0119133, 0.0119818
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0052423, 0.0052347

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 172

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013837, upper bound: 0.0014023
time: 1.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014056, upper bound: 0.0013854
time: 1.52 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 4.26 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.26
Output dim: 2, lower bound: -0.0013855, upper bound: 0.0014056
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.26
Output dim: 2, lower bound: -0.0014023, upper bound: 0.0013837
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.26
Output dim: 2, lower bound: -0.0013908, upper bound: 0.0013941
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.26
Output dim: 2, lower bound: -0.0014121, upper bound: 0.0013770
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.26
Output dim: 2, lower bound: -0.0013769, upper bound: 0.0014123
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.26
Output dim: 2, lower bound: -0.0013940, upper bound: 0.0013908
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.26
Output dim: 2, lower bound: -0.0013835, upper bound: 0.0014022
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.26
Output dim: 2, lower bound: -0.0014047, upper bound: 0.0013854
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.26
Output dim: 2, lower bound: -0.0013854, upper bound: 0.0014048
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.26
Output dim: 2, lower bound: -0.0014023, upper bound: 0.0013835
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.26
Output dim: 2, lower bound: -0.0013908, upper bound: 0.0013940
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.26
Output dim: 2, lower bound: -0.0014123, upper bound: 0.0013770
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.26
Output dim: 2, lower bound: -0.0013770, upper bound: 0.0014121
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.26
Output dim: 2, lower bound: -0.0013941, upper bound: 0.0013908
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.26
Output dim: 2, lower bound: -0.0013837, upper bound: 0.0014023
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.26
Output dim: 2, lower bound: -0.0014056, upper bound: 0.0013854

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0112368, 0.0114202
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051880, 0.0051679

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 182

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013812, upper bound: 0.0014009
time: 1.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013808, upper bound: 0.0014009
time: 1.91 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0114656, 0.0111683
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051604, 0.0051930

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 182

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013977, upper bound: 0.0013792
time: 1.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013975, upper bound: 0.0013793
time: 1.55 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0113026, 0.0113355
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051787, 0.0051751

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 182

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013863, upper bound: 0.0013894
time: 1.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013862, upper bound: 0.0013897
time: 1.64 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0115436, 0.0110879
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051515, 0.0052016

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 182

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014073, upper bound: 0.0013725
time: 1.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014073, upper bound: 0.0013727
time: 1.77 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0110836, 0.0115497
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0052022, 0.0051511

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 182

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013727, upper bound: 0.0014075
time: 1.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013725, upper bound: 0.0014075
time: 1.68 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0113272, 0.0113067
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051755, 0.0051778

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 182

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013896, upper bound: 0.0013862
time: 1.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013893, upper bound: 0.0013864
time: 1.69 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0111623, 0.0114695
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051934, 0.0051597

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 182

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013791, upper bound: 0.0013975
time: 1.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013790, upper bound: 0.0013977
time: 1.68 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0114114, 0.0112390
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051681, 0.0051870

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 182

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014001, upper bound: 0.0013808
time: 1.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014000, upper bound: 0.0013811
time: 1.70 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0112390, 0.0114114
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051870, 0.0051681

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 182

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013811, upper bound: 0.0014000
time: 1.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013808, upper bound: 0.0014001
time: 1.99 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0114695, 0.0111622
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051597, 0.0051934

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 182

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013977, upper bound: 0.0013790
time: 1.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013975, upper bound: 0.0013791
time: 2.04 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0113067, 0.0113272
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051778, 0.0051755

Time for backsubstitution: 1.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 182

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013864, upper bound: 0.0013893
time: 1.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013862, upper bound: 0.0013896
time: 1.60 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0115497, 0.0110837
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051511, 0.0052022

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 182

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014075, upper bound: 0.0013725
time: 1.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014075, upper bound: 0.0013727
time: 1.77 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0110879, 0.0115436
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0052016, 0.0051515

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 182

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013727, upper bound: 0.0014073
time: 1.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013725, upper bound: 0.0014073
time: 2.14 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0113355, 0.0113026
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051751, 0.0051787

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 182

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013897, upper bound: 0.0013862
time: 1.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013894, upper bound: 0.0013864
time: 1.90 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0111683, 0.0114656
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051930, 0.0051604

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 182

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013793, upper bound: 0.0013975
time: 1.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013792, upper bound: 0.0013977
time: 1.55 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0114202, 0.0112368
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051679, 0.0051880

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 182

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014009, upper bound: 0.0013808
time: 1.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014009, upper bound: 0.0013812
time: 2.01 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 5.30 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.30
Output dim: 2, lower bound: -0.0013812, upper bound: 0.0014009
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.30
Output dim: 2, lower bound: -0.0013808, upper bound: 0.0014009
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.30
Output dim: 2, lower bound: -0.0013977, upper bound: 0.0013792
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.30
Output dim: 2, lower bound: -0.0013975, upper bound: 0.0013793
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.30
Output dim: 2, lower bound: -0.0013863, upper bound: 0.0013894
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.30
Output dim: 2, lower bound: -0.0013862, upper bound: 0.0013897
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.30
Output dim: 2, lower bound: -0.0014073, upper bound: 0.0013725
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.30
Output dim: 2, lower bound: -0.0014073, upper bound: 0.0013727
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.30
Output dim: 2, lower bound: -0.0013727, upper bound: 0.0014075
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.30
Output dim: 2, lower bound: -0.0013725, upper bound: 0.0014075
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.30
Output dim: 2, lower bound: -0.0013896, upper bound: 0.0013862
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.30
Output dim: 2, lower bound: -0.0013893, upper bound: 0.0013864
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.30
Output dim: 2, lower bound: -0.0013791, upper bound: 0.0013975
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.30
Output dim: 2, lower bound: -0.0013790, upper bound: 0.0013977
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.30
Output dim: 2, lower bound: -0.0014001, upper bound: 0.0013808
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.30
Output dim: 2, lower bound: -0.0014000, upper bound: 0.0013811
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.30
Output dim: 2, lower bound: -0.0013811, upper bound: 0.0014000
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.30
Output dim: 2, lower bound: -0.0013808, upper bound: 0.0014001
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.30
Output dim: 2, lower bound: -0.0013977, upper bound: 0.0013790
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.30
Output dim: 2, lower bound: -0.0013975, upper bound: 0.0013791
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.30
Output dim: 2, lower bound: -0.0013864, upper bound: 0.0013893
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.30
Output dim: 2, lower bound: -0.0013862, upper bound: 0.0013896
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.30
Output dim: 2, lower bound: -0.0014075, upper bound: 0.0013725
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.30
Output dim: 2, lower bound: -0.0014075, upper bound: 0.0013727
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.30
Output dim: 2, lower bound: -0.0013727, upper bound: 0.0014073
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.30
Output dim: 2, lower bound: -0.0013725, upper bound: 0.0014073
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.30
Output dim: 2, lower bound: -0.0013897, upper bound: 0.0013862
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.30
Output dim: 2, lower bound: -0.0013894, upper bound: 0.0013864
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.30
Output dim: 2, lower bound: -0.0013793, upper bound: 0.0013975
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.30
Output dim: 2, lower bound: -0.0013792, upper bound: 0.0013977
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.30
Output dim: 2, lower bound: -0.0014009, upper bound: 0.0013808
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.30
Output dim: 2, lower bound: -0.0014009, upper bound: 0.0013812

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0111799, 0.0113505
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051804, 0.0051617

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013795, upper bound: 0.0013992
time: 1.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013788, upper bound: 0.0013993
time: 1.83 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0111671, 0.0114202
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051880, 0.0051603

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013792, upper bound: 0.0013992
time: 1.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013785, upper bound: 0.0013993
time: 1.85 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0113964, 0.0110986
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051528, 0.0051855

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013961, upper bound: 0.0013771
time: 1.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013957, upper bound: 0.0013776
time: 1.95 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0113959, 0.0111683
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051604, 0.0051854

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013959, upper bound: 0.0013772
time: 1.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013956, upper bound: 0.0013778
time: 1.80 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0112438, 0.0112658
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051711, 0.0051687

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013848, upper bound: 0.0013876
time: 1.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013841, upper bound: 0.0013878
time: 2.05 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0112329, 0.0113355
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051787, 0.0051675

Time for backsubstitution: 1.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013846, upper bound: 0.0013879
time: 1.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013838, upper bound: 0.0013881
time: 1.64 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0114739, 0.0110182
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051439, 0.0051940

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014056, upper bound: 0.0013699
time: 1.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014056, upper bound: 0.0013710
time: 1.65 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0114739, 0.0110879
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051515, 0.0051940

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014057, upper bound: 0.0013700
time: 1.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014056, upper bound: 0.0013711
time: 1.88 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0110248, 0.0114800
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051947, 0.0051447

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013699, upper bound: 0.0014058
time: 1.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013699, upper bound: 0.0014059
time: 1.83 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0110139, 0.0115497
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0052022, 0.0051435

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013710, upper bound: 0.0014058
time: 1.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013698, upper bound: 0.0014058
time: 1.64 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0112613, 0.0112370
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051680, 0.0051706

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013880, upper bound: 0.0013839
time: 1.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013877, upper bound: 0.0013846
time: 1.90 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0112574, 0.0113067
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051755, 0.0051702

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013710, upper bound: 0.0013841
time: 1.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013875, upper bound: 0.0013848
time: 1.99 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0110982, 0.0113997
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051858, 0.0051527

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013776, upper bound: 0.0013956
time: 1.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013699, upper bound: 0.0013959
time: 1.90 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0110925, 0.0114695
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051934, 0.0051521

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013774, upper bound: 0.0013957
time: 1.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013699, upper bound: 0.0013961
time: 1.94 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0113433, 0.0111693
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051605, 0.0051796

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013985, upper bound: 0.0013783
time: 1.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013984, upper bound: 0.0013791
time: 1.69 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0113417, 0.0112390
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051681, 0.0051795

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013984, upper bound: 0.0013788
time: 1.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013983, upper bound: 0.0013795
time: 1.71 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0111816, 0.0113417
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051795, 0.0051619

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013795, upper bound: 0.0013983
time: 1.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013788, upper bound: 0.0013984
time: 1.94 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0111693, 0.0114114
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051870, 0.0051605

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013699, upper bound: 0.0013984
time: 1.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013784, upper bound: 0.0013985
time: 1.63 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0114000, 0.0110925
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051521, 0.0051859

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013960, upper bound: 0.0013769
time: 1.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013957, upper bound: 0.0013774
time: 1.96 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0113998, 0.0111622
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051597, 0.0051858

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013959, upper bound: 0.0013770
time: 1.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013955, upper bound: 0.0013775
time: 1.76 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0112475, 0.0112575
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051702, 0.0051691

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013848, upper bound: 0.0013875
time: 1.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013841, upper bound: 0.0013877
time: 1.70 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0112370, 0.0113272
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051778, 0.0051680

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013845, upper bound: 0.0013877
time: 1.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013839, upper bound: 0.0013881
time: 1.77 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0114791, 0.0110139
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051435, 0.0051946

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014058, upper bound: 0.0013698
time: 1.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014058, upper bound: 0.0013709
time: 1.59 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0114800, 0.0110837
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051511, 0.0051947

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014058, upper bound: 0.0013699
time: 1.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014058, upper bound: 0.0013711
time: 1.67 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0110308, 0.0114739
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051940, 0.0051453

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013711, upper bound: 0.0014056
time: 1.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013699, upper bound: 0.0014056
time: 1.83 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0110182, 0.0115436
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0052016, 0.0051439

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013709, upper bound: 0.0014056
time: 1.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013699, upper bound: 0.0014057
time: 1.67 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0112686, 0.0112329
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051675, 0.0051714

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013881, upper bound: 0.0013840
time: 1.90 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013879, upper bound: 0.0013846
time: 1.97 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0112658, 0.0113026
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051751, 0.0051711

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013877, upper bound: 0.0013841
time: 1.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013876, upper bound: 0.0013848
time: 1.83 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0111048, 0.0113959
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051854, 0.0051535

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013777, upper bound: 0.0013956
time: 1.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013772, upper bound: 0.0013959
time: 1.78 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0110986, 0.0114656
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051930, 0.0051528

Time for backsubstitution: 1.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013775, upper bound: 0.0013957
time: 1.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013770, upper bound: 0.0013961
time: 1.70 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0113508, 0.0111671
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051603, 0.0051805

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013992, upper bound: 0.0013786
time: 1.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013991, upper bound: 0.0013792
time: 1.80 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0113505, 0.0112368
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051679, 0.0051804

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013992, upper bound: 0.0013788
time: 1.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013992, upper bound: 0.0013796
time: 1.80 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 4.89 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 2, lower bound: -0.0013795, upper bound: 0.0013992
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 2, lower bound: -0.0013788, upper bound: 0.0013993
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 2, lower bound: -0.0013792, upper bound: 0.0013992
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 2, lower bound: -0.0013785, upper bound: 0.0013993
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 2, lower bound: -0.0013961, upper bound: 0.0013771
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 2, lower bound: -0.0013957, upper bound: 0.0013776
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 2, lower bound: -0.0013959, upper bound: 0.0013772
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 2, lower bound: -0.0013956, upper bound: 0.0013778
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 2, lower bound: -0.0013848, upper bound: 0.0013876
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 2, lower bound: -0.0013841, upper bound: 0.0013878
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 2, lower bound: -0.0013846, upper bound: 0.0013879
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 2, lower bound: -0.0013838, upper bound: 0.0013881
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 2, lower bound: -0.0014056, upper bound: 0.0013699
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 2, lower bound: -0.0014056, upper bound: 0.0013710
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 2, lower bound: -0.0014057, upper bound: 0.0013700
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 2, lower bound: -0.0014056, upper bound: 0.0013711
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 2, lower bound: -0.0013699, upper bound: 0.0014058
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 2, lower bound: -0.0013699, upper bound: 0.0014059
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 2, lower bound: -0.0013710, upper bound: 0.0014058
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 2, lower bound: -0.0013698, upper bound: 0.0014058
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 2, lower bound: -0.0013880, upper bound: 0.0013839
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 2, lower bound: -0.0013877, upper bound: 0.0013846
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 2, lower bound: -0.0013710, upper bound: 0.0013841
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 2, lower bound: -0.0013875, upper bound: 0.0013848
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 2, lower bound: -0.0013776, upper bound: 0.0013956
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 2, lower bound: -0.0013699, upper bound: 0.0013959
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 2, lower bound: -0.0013774, upper bound: 0.0013957
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 2, lower bound: -0.0013699, upper bound: 0.0013961
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 2, lower bound: -0.0013985, upper bound: 0.0013783
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 2, lower bound: -0.0013984, upper bound: 0.0013791
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 2, lower bound: -0.0013984, upper bound: 0.0013788
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 2, lower bound: -0.0013983, upper bound: 0.0013795
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 2, lower bound: -0.0013795, upper bound: 0.0013983
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 2, lower bound: -0.0013788, upper bound: 0.0013984
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 2, lower bound: -0.0013699, upper bound: 0.0013984
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 2, lower bound: -0.0013784, upper bound: 0.0013985
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 2, lower bound: -0.0013960, upper bound: 0.0013769
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 2, lower bound: -0.0013957, upper bound: 0.0013774
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 2, lower bound: -0.0013959, upper bound: 0.0013770
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 2, lower bound: -0.0013955, upper bound: 0.0013775
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 2, lower bound: -0.0013848, upper bound: 0.0013875
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 2, lower bound: -0.0013841, upper bound: 0.0013877
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 2, lower bound: -0.0013845, upper bound: 0.0013877
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 2, lower bound: -0.0013839, upper bound: 0.0013881
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 2, lower bound: -0.0014058, upper bound: 0.0013698
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 2, lower bound: -0.0014058, upper bound: 0.0013709
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 2, lower bound: -0.0014058, upper bound: 0.0013699
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 2, lower bound: -0.0014058, upper bound: 0.0013711
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 2, lower bound: -0.0013711, upper bound: 0.0014056
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 2, lower bound: -0.0013699, upper bound: 0.0014056
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 2, lower bound: -0.0013709, upper bound: 0.0014056
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 2, lower bound: -0.0013699, upper bound: 0.0014057
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 2, lower bound: -0.0013881, upper bound: 0.0013840
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 2, lower bound: -0.0013879, upper bound: 0.0013846
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 2, lower bound: -0.0013877, upper bound: 0.0013841
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 2, lower bound: -0.0013876, upper bound: 0.0013848
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 2, lower bound: -0.0013777, upper bound: 0.0013956
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 2, lower bound: -0.0013772, upper bound: 0.0013959
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 2, lower bound: -0.0013775, upper bound: 0.0013957
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 2, lower bound: -0.0013770, upper bound: 0.0013961
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 2, lower bound: -0.0013992, upper bound: 0.0013786
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 2, lower bound: -0.0013991, upper bound: 0.0013792
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 2, lower bound: -0.0013992, upper bound: 0.0013788
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 2, lower bound: -0.0013992, upper bound: 0.0013796

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0110792, 0.0112433
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051718, 0.0051538

Time for backsubstitution: 1.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 248

Time for candidate selection: 2.31 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012342, upper bound: 0.0012574
time: 1.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012342, upper bound: 0.0012574
time: 1.39 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0110727, 0.0112392
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051714, 0.0051531

Time for backsubstitution: 1.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 248

Time for candidate selection: 2.07 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012342, upper bound: 0.0012570
time: 1.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012342, upper bound: 0.0012570
time: 1.47 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0110656, 0.0113123
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051793, 0.0051523

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 248

Time for candidate selection: 2.10 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012342, upper bound: 0.0012574
time: 1.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012342, upper bound: 0.0012574
time: 1.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0110599, 0.0113082
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051789, 0.0051517

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 248

Time for candidate selection: 2.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012342, upper bound: 0.0012570
time: 1.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012342, upper bound: 0.0012570
time: 1.38 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0112915, 0.0109914
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051442, 0.0051771

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 248

Time for candidate selection: 2.06 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012501, upper bound: 0.0012389
time: 1.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012501, upper bound: 0.0012389
time: 1.33 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0112892, 0.0109940
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051445, 0.0051769

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 248

Time for candidate selection: 1.99 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012504, upper bound: 0.0012389
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012504, upper bound: 0.0012389
time: 1.16 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0112887, 0.0110604
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051517, 0.0051768

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 248

Time for candidate selection: 2.03 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012501, upper bound: 0.0012389
time: 1.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012501, upper bound: 0.0012389
time: 1.54 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0112887, 0.0110631
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051520, 0.0051768

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 248

Time for candidate selection: 2.02 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012504, upper bound: 0.0012389
time: 1.36 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012504, upper bound: 0.0012389
time: 1.36 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0111411, 0.0111586
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051625, 0.0051606

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 248

Time for candidate selection: 2.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012406, upper bound: 0.0012495
time: 1.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012406, upper bound: 0.0012495
time: 1.69 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0111366, 0.0111565
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051623, 0.0051601

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 248

Time for candidate selection: 2.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012406, upper bound: 0.0012494
time: 1.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012406, upper bound: 0.0012494
time: 1.15 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0111309, 0.0112276
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051700, 0.0051595

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 248

Time for candidate selection: 1.97 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012406, upper bound: 0.0012495
time: 1.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012406, upper bound: 0.0012495
time: 1.91 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0111257, 0.0112255
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051698, 0.0051589

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 248

Time for candidate selection: 2.10 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012406, upper bound: 0.0012494
time: 1.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012406, upper bound: 0.0012494
time: 1.56 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0113675, 0.0109110
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051354, 0.0051855

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 248

Time for candidate selection: 1.99 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012584, upper bound: 0.0012327
time: 1.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012584, upper bound: 0.0012327
time: 1.16 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0113667, 0.0109158
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051359, 0.0051854

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 248

Time for candidate selection: 2.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012586, upper bound: 0.0012327
time: 1.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012586, upper bound: 0.0012327
time: 1.45 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0113661, 0.0109800
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051429, 0.0051853

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 248

Time for candidate selection: 2.03 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012584, upper bound: 0.0012327
time: 1.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012584, upper bound: 0.0012327
time: 1.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0113667, 0.0109848
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051434, 0.0051854

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 248

Time for candidate selection: 1.98 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012586, upper bound: 0.0012327
time: 1.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012586, upper bound: 0.0012327
time: 1.89 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0109237, 0.0113728
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051861, 0.0051368

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 248

Time for candidate selection: 2.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012327, upper bound: 0.0012588
time: 1.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012327, upper bound: 0.0012588
time: 1.37 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0109176, 0.0113696
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051857, 0.0051361

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 248

Time for candidate selection: 2.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012327, upper bound: 0.0012585
time: 1.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012327, upper bound: 0.0012585
time: 1.49 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0109138, 0.0114419
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051936, 0.0051357

Time for backsubstitution: 1.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 248

Time for candidate selection: 2.26 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012327, upper bound: 0.0012588
time: 1.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012327, upper bound: 0.0012588
time: 1.49 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0109067, 0.0114387
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051932, 0.0051349

Time for backsubstitution: 1.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 248

Time for candidate selection: 2.26 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012327, upper bound: 0.0012585
time: 1.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012327, upper bound: 0.0012585
time: 1.31 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0111532, 0.0111298
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051594, 0.0051620

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 248

Time for candidate selection: 1.99 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012493, upper bound: 0.0012406
time: 1.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012493, upper bound: 0.0012406
time: 1.18 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0111541, 0.0111319
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051596, 0.0051620

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 248

Time for candidate selection: 1.98 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012495, upper bound: 0.0012406
time: 1.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012495, upper bound: 0.0012406
time: 1.58 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0111491, 0.0111988
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051669, 0.0051615

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 248

Time for candidate selection: 1.99 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012493, upper bound: 0.0012406
time: 1.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012493, upper bound: 0.0012406
time: 1.46 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0111503, 0.0112009
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051671, 0.0051616

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 248

Time for candidate selection: 2.01 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012495, upper bound: 0.0012406
time: 1.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012495, upper bound: 0.0012406
time: 1.38 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0109955, 0.0112926
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051773, 0.0051446

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 248

Time for candidate selection: 2.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012387, upper bound: 0.0012504
time: 1.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012387, upper bound: 0.0012504
time: 1.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0109910, 0.0112922
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051772, 0.0051441

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 248

Time for candidate selection: 2.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012387, upper bound: 0.0012502
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012387, upper bound: 0.0012502
time: 1.14 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0109899, 0.0113616
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051848, 0.0051440

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 248

Time for candidate selection: 1.95 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012387, upper bound: 0.0012504
time: 1.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012387, upper bound: 0.0012504
time: 1.39 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0109853, 0.0113612
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051847, 0.0051435

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 248

Time for candidate selection: 2.02 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012387, upper bound: 0.0012502
time: 1.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012387, upper bound: 0.0012502
time: 1.18 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0112340, 0.0110621
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051520, 0.0051708

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 248

Time for candidate selection: 2.07 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012569, upper bound: 0.0012342
time: 1.05 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012569, upper bound: 0.0012342
time: 1.04 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0112361, 0.0110657
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051523, 0.0051711

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 248

Time for candidate selection: 2.34 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012572, upper bound: 0.0012342
time: 1.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012572, upper bound: 0.0012342
time: 1.40 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0112306, 0.0111311
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051595, 0.0051705

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 248

Time for candidate selection: 2.04 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012569, upper bound: 0.0012342
time: 1.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012569, upper bound: 0.0012342
time: 1.53 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0112345, 0.0111347
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051598, 0.0051709

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 248

Time for candidate selection: 2.02 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012572, upper bound: 0.0012342
time: 1.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012572, upper bound: 0.0012342
time: 1.38 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0110784, 0.0112345
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051709, 0.0051537

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 248

Time for candidate selection: 1.99 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012342, upper bound: 0.0012572
time: 1.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012342, upper bound: 0.0012572
time: 1.40 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0110744, 0.0112306
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051705, 0.0051533

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 248

Time for candidate selection: 1.99 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012342, upper bound: 0.0012569
time: 1.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012342, upper bound: 0.0012569
time: 1.29 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0110657, 0.0113035
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051784, 0.0051523

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 248

Time for candidate selection: 2.02 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012342, upper bound: 0.0012572
time: 1.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012342, upper bound: 0.0012572
time: 1.16 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0110621, 0.0112996
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051780, 0.0051520

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 248

Time for candidate selection: 2.03 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012342, upper bound: 0.0012569
time: 1.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012342, upper bound: 0.0012569
time: 1.33 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0112942, 0.0109853
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051435, 0.0051774

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 248

Time for candidate selection: 2.07 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012502, upper bound: 0.0012387
time: 1.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012502, upper bound: 0.0012387
time: 1.37 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0112928, 0.0109899
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051440, 0.0051773

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 248

Time for candidate selection: 2.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012504, upper bound: 0.0012387
time: 1.05 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012504, upper bound: 0.0012387
time: 1.06 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0112922, 0.0110544
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051510, 0.0051772

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 248

Time for candidate selection: 2.04 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012502, upper bound: 0.0012387
time: 1.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012502, upper bound: 0.0012387
time: 1.58 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0112926, 0.0110589
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051515, 0.0051773

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 248

Time for candidate selection: 2.01 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012504, upper bound: 0.0012387
time: 1.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012504, upper bound: 0.0012387
time: 1.34 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0111427, 0.0111502
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051616, 0.0051608

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 248

Time for candidate selection: 2.05 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012406, upper bound: 0.0012495
time: 1.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012406, upper bound: 0.0012495
time: 1.41 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0111403, 0.0111491
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051615, 0.0051605

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 248

Time for candidate selection: 2.02 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012406, upper bound: 0.0012493
time: 1.00 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012406, upper bound: 0.0012493
time: 1.00 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0111319, 0.0112193
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051691, 0.0051596

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 248

Time for candidate selection: 2.04 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012406, upper bound: 0.0012495
time: 1.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012406, upper bound: 0.0012495
time: 1.55 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0111298, 0.0112181
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051690, 0.0051594

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 248

Time for candidate selection: 2.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012406, upper bound: 0.0012493
time: 1.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012406, upper bound: 0.0012493
time: 1.35 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0113713, 0.0109067
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051349, 0.0051859

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 248

Time for candidate selection: 2.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012585, upper bound: 0.0012327
time: 1.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012585, upper bound: 0.0012327
time: 1.27 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0113719, 0.0109138
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051357, 0.0051860

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 248

Time for candidate selection: 2.03 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012588, upper bound: 0.0012327
time: 1.08 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012588, upper bound: 0.0012327
time: 1.10 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0113696, 0.0109758
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051424, 0.0051857

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 248

Time for candidate selection: 2.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012585, upper bound: 0.0012327
time: 1.09 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012585, upper bound: 0.0012327
time: 1.04 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0113728, 0.0109828
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051432, 0.0051861

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 248

Time for candidate selection: 2.02 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012588, upper bound: 0.0012327
time: 1.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012588, upper bound: 0.0012327
time: 1.37 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0109260, 0.0113667
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051854, 0.0051370

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 248

Time for candidate selection: 2.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012327, upper bound: 0.0012586
time: 1.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012327, upper bound: 0.0012586
time: 1.05 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0109236, 0.0113661
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051853, 0.0051367

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 248

Time for candidate selection: 2.01 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012327, upper bound: 0.0012584
time: 1.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012327, upper bound: 0.0012584
time: 1.42 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0109158, 0.0114357
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051929, 0.0051359

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 248

Time for candidate selection: 2.03 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012327, upper bound: 0.0012586
time: 1.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012327, upper bound: 0.0012586
time: 1.35 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0109110, 0.0114352
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051928, 0.0051354

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 248

Time for candidate selection: 2.01 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012327, upper bound: 0.0012584
time: 1.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012327, upper bound: 0.0012584
time: 1.43 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 5.95 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012342, upper bound: 0.0012574
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012342, upper bound: 0.0012574
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012342, upper bound: 0.0012570
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012342, upper bound: 0.0012570
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012342, upper bound: 0.0012574
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012342, upper bound: 0.0012574
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012342, upper bound: 0.0012570
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012342, upper bound: 0.0012570
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012501, upper bound: 0.0012389
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012501, upper bound: 0.0012389
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012504, upper bound: 0.0012389
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012504, upper bound: 0.0012389
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012501, upper bound: 0.0012389
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012501, upper bound: 0.0012389
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012504, upper bound: 0.0012389
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012504, upper bound: 0.0012389
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012406, upper bound: 0.0012495
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012406, upper bound: 0.0012495
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012406, upper bound: 0.0012494
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012406, upper bound: 0.0012494
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012406, upper bound: 0.0012495
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012406, upper bound: 0.0012495
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012406, upper bound: 0.0012494
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012406, upper bound: 0.0012494
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012584, upper bound: 0.0012327
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012584, upper bound: 0.0012327
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012586, upper bound: 0.0012327
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012586, upper bound: 0.0012327
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012584, upper bound: 0.0012327
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012584, upper bound: 0.0012327
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012586, upper bound: 0.0012327
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012586, upper bound: 0.0012327
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012327, upper bound: 0.0012588
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012327, upper bound: 0.0012588
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012327, upper bound: 0.0012585
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012327, upper bound: 0.0012585
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012327, upper bound: 0.0012588
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012327, upper bound: 0.0012588
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012327, upper bound: 0.0012585
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012327, upper bound: 0.0012585
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012493, upper bound: 0.0012406
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012493, upper bound: 0.0012406
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012495, upper bound: 0.0012406
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012495, upper bound: 0.0012406
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012493, upper bound: 0.0012406
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012493, upper bound: 0.0012406
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012495, upper bound: 0.0012406
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012495, upper bound: 0.0012406
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012387, upper bound: 0.0012504
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012387, upper bound: 0.0012504
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012387, upper bound: 0.0012502
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012387, upper bound: 0.0012502
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012387, upper bound: 0.0012504
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012387, upper bound: 0.0012504
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012387, upper bound: 0.0012502
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012387, upper bound: 0.0012502
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012569, upper bound: 0.0012342
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012569, upper bound: 0.0012342
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012572, upper bound: 0.0012342
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012572, upper bound: 0.0012342
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012569, upper bound: 0.0012342
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012569, upper bound: 0.0012342
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012572, upper bound: 0.0012342
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012572, upper bound: 0.0012342
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012342, upper bound: 0.0012572
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012342, upper bound: 0.0012572
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012342, upper bound: 0.0012569
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012342, upper bound: 0.0012569
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012342, upper bound: 0.0012572
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012342, upper bound: 0.0012572
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012342, upper bound: 0.0012569
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012342, upper bound: 0.0012569
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012502, upper bound: 0.0012387
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012502, upper bound: 0.0012387
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012504, upper bound: 0.0012387
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012504, upper bound: 0.0012387
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012502, upper bound: 0.0012387
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012502, upper bound: 0.0012387
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012504, upper bound: 0.0012387
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012504, upper bound: 0.0012387
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012406, upper bound: 0.0012495
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012406, upper bound: 0.0012495
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012406, upper bound: 0.0012493
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012406, upper bound: 0.0012493
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012406, upper bound: 0.0012495
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012406, upper bound: 0.0012495
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012406, upper bound: 0.0012493
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012406, upper bound: 0.0012493
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012585, upper bound: 0.0012327
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012585, upper bound: 0.0012327
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012588, upper bound: 0.0012327
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012588, upper bound: 0.0012327
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012585, upper bound: 0.0012327
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012585, upper bound: 0.0012327
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012588, upper bound: 0.0012327
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012588, upper bound: 0.0012327
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012327, upper bound: 0.0012586
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012327, upper bound: 0.0012586
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012327, upper bound: 0.0012584
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012327, upper bound: 0.0012584
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012327, upper bound: 0.0012586
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012327, upper bound: 0.0012586
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012327, upper bound: 0.0012584
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.95
Output dim: 2, lower bound: -0.0012327, upper bound: 0.0012584
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 2, lower bound: -0.0013881, upper bound: 0.0013840
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 2, lower bound: -0.0013879, upper bound: 0.0013846
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 2, lower bound: -0.0013877, upper bound: 0.0013841
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 2, lower bound: -0.0013876, upper bound: 0.0013848
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 2, lower bound: -0.0013777, upper bound: 0.0013956
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 2, lower bound: -0.0013772, upper bound: 0.0013959
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 2, lower bound: -0.0013775, upper bound: 0.0013957
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 2, lower bound: -0.0013770, upper bound: 0.0013961
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 2, lower bound: -0.0013992, upper bound: 0.0013786
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 2, lower bound: -0.0013991, upper bound: 0.0013792
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 2, lower bound: -0.0013992, upper bound: 0.0013788
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 2, lower bound: -0.0013992, upper bound: 0.0013796

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 3.49 + 599.11 = 602.60 seconds
