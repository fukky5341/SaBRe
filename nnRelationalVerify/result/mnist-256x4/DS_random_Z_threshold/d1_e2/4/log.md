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
execution time: IAR + RelationalAnalysis = 0.93 + 2.46 = 3.39 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -0.0014289, upper bound: 0.0014289

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014267, upper bound: 0.0014270
time: 1.70 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014269, upper bound: 0.0014268
time: 1.13 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 2.84 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 2.84
Output dim: 2, lower bound: -0.0014267, upper bound: 0.0014270
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 2.84
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

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014194, upper bound: 0.0014127
time: 1.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014119, upper bound: 0.0014195
time: 1.58 seconds

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

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014196, upper bound: 0.0014119
time: 1.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014127, upper bound: 0.0014194
time: 1.67 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 3.91 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 3.91
Output dim: 2, lower bound: -0.0014194, upper bound: 0.0014127
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 3.91
Output dim: 2, lower bound: -0.0014119, upper bound: 0.0014195
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 3.91
Output dim: 2, lower bound: -0.0014196, upper bound: 0.0014119
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 3.91
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

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 172

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0011243, upper bound: 0.0011224
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0011243, upper bound: 0.0011224
time: 0.75 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 172

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013883, upper bound: 0.0014166
time: 1.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014089, upper bound: 0.0013957
time: 1.94 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014053, upper bound: 0.0014077
time: 1.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014152, upper bound: 0.0013971
time: 1.57 seconds

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

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013971, upper bound: 0.0014150
time: 1.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014085, upper bound: 0.0014054
time: 1.51 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 3.81 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 3.81
Output dim: 2, lower bound: -0.0011243, upper bound: 0.0011224
DS_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 3.81
Output dim: 2, lower bound: -0.0011243, upper bound: 0.0011224
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.81
Output dim: 2, lower bound: -0.0013883, upper bound: 0.0014166
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.81
Output dim: 2, lower bound: -0.0014089, upper bound: 0.0013957
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.81
Output dim: 2, lower bound: -0.0014053, upper bound: 0.0014077
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.81
Output dim: 2, lower bound: -0.0014152, upper bound: 0.0013971
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.81
Output dim: 2, lower bound: -0.0013971, upper bound: 0.0014150
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.81
Output dim: 2, lower bound: -0.0014085, upper bound: 0.0014054

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0113797, 0.0117655
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0052243, 0.0051819

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013868, upper bound: 0.0014149
time: 1.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013860, upper bound: 0.0014149
time: 1.70 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0116232, 0.0115351
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051990, 0.0052087

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0011039, upper bound: 0.0011035
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0011039, upper bound: 0.0011035
time: 0.87 seconds

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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

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
time: 1.77 seconds

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

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0010964, upper bound: 0.0010919
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0010964, upper bound: 0.0010919
time: 0.69 seconds

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

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013955, upper bound: 0.0014133
time: 1.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013953, upper bound: 0.0014134
time: 1.60 seconds

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

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 182

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014039, upper bound: 0.0014006
time: 1.92 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014038, upper bound: 0.0014008
time: 1.69 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 4.48 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.48
Output dim: 2, lower bound: -0.0013868, upper bound: 0.0014149
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.48
Output dim: 2, lower bound: -0.0013860, upper bound: 0.0014149
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 4.48
Output dim: 2, lower bound: -0.0011039, upper bound: 0.0011035
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 4.48
Output dim: 2, lower bound: -0.0011039, upper bound: 0.0011035
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.48
Output dim: 2, lower bound: -0.0013854, upper bound: 0.0014048
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.48
Output dim: 2, lower bound: -0.0014023, upper bound: 0.0013835
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 4.48
Output dim: 2, lower bound: -0.0010964, upper bound: 0.0010919
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 4.48
Output dim: 2, lower bound: -0.0010964, upper bound: 0.0010919
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.48
Output dim: 2, lower bound: -0.0013955, upper bound: 0.0014133
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.48
Output dim: 2, lower bound: -0.0013953, upper bound: 0.0014134
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.48
Output dim: 2, lower bound: -0.0014039, upper bound: 0.0014006
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.48
Output dim: 2, lower bound: -0.0014038, upper bound: 0.0014008

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0112828, 0.0116616
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0052161, 0.0051745

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013754, upper bound: 0.0014106
time: 1.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013819, upper bound: 0.0014003
time: 1.83 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0112758, 0.0116612
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0052161, 0.0051737

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 182

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013816, upper bound: 0.0014102
time: 1.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013815, upper bound: 0.0014101
time: 1.59 seconds

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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 182

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013811, upper bound: 0.0014000
time: 1.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013808, upper bound: 0.0014001
time: 1.97 seconds

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

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014006, upper bound: 0.0013814
time: 1.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014003, upper bound: 0.0013820
time: 1.58 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0117265, 0.0119364
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0052412, 0.0052182

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 172

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 182

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013911, upper bound: 0.0014086
time: 1.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013908, upper bound: 0.0014086
time: 1.76 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0117217, 0.0119417
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0052418, 0.0052177

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 172

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 182

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013909, upper bound: 0.0014087
time: 1.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013906, upper bound: 0.0014086
time: 1.68 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0118456, 0.0119079
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0052347, 0.0052279

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 172

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014022, upper bound: 0.0013987
time: 1.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014022, upper bound: 0.0013990
time: 1.47 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0118394, 0.0119818
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0052423, 0.0052272

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 172

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014022, upper bound: 0.0013987
time: 1.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014021, upper bound: 0.0013991
time: 1.46 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 3.79 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 2, lower bound: -0.0013754, upper bound: 0.0014106
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 2, lower bound: -0.0013819, upper bound: 0.0014003
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 2, lower bound: -0.0013816, upper bound: 0.0014102
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 2, lower bound: -0.0013815, upper bound: 0.0014101
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 2, lower bound: -0.0013811, upper bound: 0.0014000
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 2, lower bound: -0.0013808, upper bound: 0.0014001
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 2, lower bound: -0.0014006, upper bound: 0.0013814
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 2, lower bound: -0.0014003, upper bound: 0.0013820
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 2, lower bound: -0.0013911, upper bound: 0.0014086
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 2, lower bound: -0.0013908, upper bound: 0.0014086
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 2, lower bound: -0.0013909, upper bound: 0.0014087
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 2, lower bound: -0.0013906, upper bound: 0.0014086
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 2, lower bound: -0.0014022, upper bound: 0.0013987
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 2, lower bound: -0.0014022, upper bound: 0.0013990
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 2, lower bound: -0.0014022, upper bound: 0.0013987
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 2, lower bound: -0.0014021, upper bound: 0.0013991

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0109828, 0.0114419
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051936, 0.0051432

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 182

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013711, upper bound: 0.0014058
time: 1.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013699, upper bound: 0.0014058
time: 1.57 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0110589, 0.0113616
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051848, 0.0051515

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0010708, upper bound: 0.0010747
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0010708, upper bound: 0.0010747
time: 0.71 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0112174, 0.0115920
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0052086, 0.0051674

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013699, upper bound: 0.0014058
time: 1.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013699, upper bound: 0.0013959
time: 1.88 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0112065, 0.0116612
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0052161, 0.0051663

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013698, upper bound: 0.0014058
time: 1.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013699, upper bound: 0.0013961
time: 1.96 seconds

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

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013795, upper bound: 0.0013983
time: 1.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013788, upper bound: 0.0013984
time: 1.93 seconds

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

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013791, upper bound: 0.0013984
time: 1.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013784, upper bound: 0.0013985
time: 1.66 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0113612, 0.0110544
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051510, 0.0051847

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 182

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013960, upper bound: 0.0013769
time: 1.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013959, upper bound: 0.0013770
time: 1.77 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0113616, 0.0110589
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051515, 0.0051848

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 182

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013957, upper bound: 0.0013774
time: 2.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013955, upper bound: 0.0013775
time: 1.88 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0116657, 0.0118654
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0052337, 0.0052118

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 172

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013711, upper bound: 0.0014056
time: 2.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013881, upper bound: 0.0013840
time: 1.85 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0116555, 0.0119364
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0052412, 0.0052107

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 172

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 172

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013709, upper bound: 0.0014056
time: 1.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013877, upper bound: 0.0013841
time: 1.84 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0116633, 0.0118706
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0052343, 0.0052115

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 172

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 172

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013699, upper bound: 0.0014056
time: 2.00 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013879, upper bound: 0.0013846
time: 2.03 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0116507, 0.0119417
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0052418, 0.0052102

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 172

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 172

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013699, upper bound: 0.0014057
time: 1.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013876, upper bound: 0.0013848
time: 1.87 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0117416, 0.0117996
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0052265, 0.0052201

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 172

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013777, upper bound: 0.0013956
time: 1.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013992, upper bound: 0.0013786
time: 1.91 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0117373, 0.0118053
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0052271, 0.0052197

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 172

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013772, upper bound: 0.0013959
time: 1.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013991, upper bound: 0.0013792
time: 1.92 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0117338, 0.0118706
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0052340, 0.0052193

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 172

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 172

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013775, upper bound: 0.0013957
time: 1.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013992, upper bound: 0.0013788
time: 1.90 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0117311, 0.0118763
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0052346, 0.0052190

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 172

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 172

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013770, upper bound: 0.0013961
time: 1.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013992, upper bound: 0.0013796
time: 1.90 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 6.49 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 2, lower bound: -0.0013711, upper bound: 0.0014058
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 2, lower bound: -0.0013699, upper bound: 0.0014058
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 6.49
Output dim: 2, lower bound: -0.0010708, upper bound: 0.0010747
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 6.49
Output dim: 2, lower bound: -0.0010708, upper bound: 0.0010747
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 2, lower bound: -0.0013699, upper bound: 0.0014058
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 2, lower bound: -0.0013699, upper bound: 0.0013959
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 2, lower bound: -0.0013698, upper bound: 0.0014058
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 2, lower bound: -0.0013699, upper bound: 0.0013961
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 2, lower bound: -0.0013795, upper bound: 0.0013983
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 2, lower bound: -0.0013788, upper bound: 0.0013984
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 2, lower bound: -0.0013791, upper bound: 0.0013984
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 2, lower bound: -0.0013784, upper bound: 0.0013985
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 2, lower bound: -0.0013960, upper bound: 0.0013769
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 2, lower bound: -0.0013959, upper bound: 0.0013770
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 2, lower bound: -0.0013957, upper bound: 0.0013774
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 2, lower bound: -0.0013955, upper bound: 0.0013775
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 2, lower bound: -0.0013711, upper bound: 0.0014056
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 2, lower bound: -0.0013881, upper bound: 0.0013840
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 2, lower bound: -0.0013709, upper bound: 0.0014056
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 2, lower bound: -0.0013877, upper bound: 0.0013841
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 2, lower bound: -0.0013699, upper bound: 0.0014056
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 2, lower bound: -0.0013879, upper bound: 0.0013846
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 2, lower bound: -0.0013699, upper bound: 0.0014057
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 2, lower bound: -0.0013876, upper bound: 0.0013848
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 2, lower bound: -0.0013777, upper bound: 0.0013956
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 2, lower bound: -0.0013992, upper bound: 0.0013786
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 2, lower bound: -0.0013772, upper bound: 0.0013959
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 2, lower bound: -0.0013991, upper bound: 0.0013792
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 2, lower bound: -0.0013775, upper bound: 0.0013957
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 2, lower bound: -0.0013992, upper bound: 0.0013788
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 2, lower bound: -0.0013770, upper bound: 0.0013961
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.49
Output dim: 2, lower bound: -0.0013992, upper bound: 0.0013796

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

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 248
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 65

Time for candidate selection: 1.92 seconds

### Candidate
type: DSZ, layer: 3, pos: 111

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013509, upper bound: 0.0013951
time: 2.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013611, upper bound: 0.0013694
time: 2.35 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 248
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 1.93 seconds

### Candidate
type: DSZ, layer: 3, pos: 245

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013704, upper bound: 0.0013932
time: 2.09 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013652, upper bound: 0.0014053
time: 2.10 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 248
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 1.95 seconds

### Candidate
type: DSZ, layer: 3, pos: 65

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013690, upper bound: 0.0013904
time: 1.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013588, upper bound: 0.0014050
time: 2.10 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 248
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 206

Time for candidate selection: 1.95 seconds

### Candidate
type: DSZ, layer: 3, pos: 187

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013499, upper bound: 0.0013898
time: 1.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013704, upper bound: 0.0013626
time: 2.17 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 248

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 111

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013518, upper bound: 0.0013952
time: 2.00 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013599, upper bound: 0.0013647
time: 2.18 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 248

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 111

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013565, upper bound: 0.0013855
time: 2.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013668, upper bound: 0.0013612
time: 2.30 seconds

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

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 248
type: DSZ, layer: 3, pos: 141

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 206

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013396, upper bound: 0.0013631
time: 1.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013396, upper bound: 0.0013631
time: 1.77 seconds

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

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 248
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 206

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 157

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013724, upper bound: 0.0013756
time: 1.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013715, upper bound: 0.0013906
time: 2.12 seconds

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

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 248
type: DSZ, layer: 3, pos: 245

Time for candidate selection: 1.96 seconds

### Candidate
type: DSZ, layer: 3, pos: 142

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012342, upper bound: 0.0012572
time: 1.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012342, upper bound: 0.0012572
time: 1.23 seconds

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

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 248
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 157

Time for candidate selection: 1.96 seconds

### Candidate
type: DSZ, layer: 3, pos: 142

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 206

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013395, upper bound: 0.0013631
time: 1.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013395, upper bound: 0.0013631
time: 1.67 seconds

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

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 248
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 142

Time for candidate selection: 1.96 seconds

### Candidate
type: DSZ, layer: 3, pos: 111

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013612, upper bound: 0.0013668
time: 1.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013855, upper bound: 0.0013565
time: 2.05 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 248
type: DSZ, layer: 3, pos: 157

Time for candidate selection: 1.94 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012502, upper bound: 0.0012387
time: 1.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012502, upper bound: 0.0012387
time: 1.54 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 248
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 141

Time for candidate selection: 1.83 seconds

### Candidate
type: DSZ, layer: 3, pos: 142

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 245

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013953, upper bound: 0.0013718
time: 1.97 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013840, upper bound: 0.0013770
time: 1.81 seconds

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

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 248

Time for candidate selection: 1.83 seconds

### Candidate
type: DSZ, layer: 3, pos: 142

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 206

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013577, upper bound: 0.0013422
time: 1.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013577, upper bound: 0.0013422
time: 1.86 seconds

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

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 248
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 65

Time for candidate selection: 1.85 seconds

### Candidate
type: DSZ, layer: 3, pos: 248

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0010622, upper bound: 0.0010719
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0010622, upper bound: 0.0010719
time: 0.91 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0111596, 0.0111257
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051589, 0.0051627

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 248
type: DSZ, layer: 3, pos: 157

Time for candidate selection: 1.83 seconds

### Candidate
type: DSZ, layer: 3, pos: 142

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 206

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0013519, upper bound: 0.0013473
time: 1.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0013519, upper bound: 0.0013473
time: 1.50 seconds

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

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 248
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 111

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 157

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013643, upper bound: 0.0013917
time: 1.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013592, upper bound: 0.0013986
time: 2.06 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0111565, 0.0111947
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051664, 0.0051623

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 248
type: DSZ, layer: 3, pos: 157

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012494, upper bound: 0.0012406
time: 1.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012494, upper bound: 0.0012406
time: 1.28 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 248
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 111

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 141

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013642, upper bound: 0.0013973
time: 1.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013642, upper bound: 0.0013969
time: 2.04 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0111614, 0.0111309
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051595, 0.0051629

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 248
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 206

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0013519, upper bound: 0.0013478
time: 1.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0013519, upper bound: 0.0013478
time: 1.26 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 248
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 142

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 141

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013642, upper bound: 0.0013973
time: 1.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013642, upper bound: 0.0013970
time: 1.92 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0111586, 0.0112000
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051670, 0.0051625

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 248
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 142

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 111

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013614, upper bound: 0.0013745
time: 2.02 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013774, upper bound: 0.0013578
time: 1.72 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0110019, 0.0112887
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051768, 0.0051453

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 248
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 187

Time for candidate selection: 1.83 seconds

### Candidate
type: DSZ, layer: 3, pos: 206

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0013425, upper bound: 0.0013566
time: 1.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0013425, upper bound: 0.0013567
time: 1.60 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0112416, 0.0110599
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051517, 0.0051717

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 248
type: DSZ, layer: 3, pos: 111

Time for candidate selection: 1.84 seconds

### Candidate
type: DSZ, layer: 3, pos: 141

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013924, upper bound: 0.0013708
time: 1.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013924, upper bound: 0.0013704
time: 1.77 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0109976, 0.0112887
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051768, 0.0051449

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 248

Time for candidate selection: 1.84 seconds

### Candidate
type: DSZ, layer: 3, pos: 245

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013767, upper bound: 0.0013828
time: 1.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013652, upper bound: 0.0013954
time: 1.63 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0112436, 0.0110656
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051523, 0.0051719

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 248
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 65

Time for candidate selection: 1.88 seconds

### Candidate
type: DSZ, layer: 3, pos: 206

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013640, upper bound: 0.0013394
time: 1.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013640, upper bound: 0.0013394
time: 1.53 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0109940, 0.0113578
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051843, 0.0051445

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 248

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 206

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0013425, upper bound: 0.0013567
time: 1.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0013425, upper bound: 0.0013568
time: 1.54 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0112392, 0.0111289
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051592, 0.0051714

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 248
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 187

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013597, upper bound: 0.0013719
time: 2.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013935, upper bound: 0.0013543
time: 1.70 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0109914, 0.0113577
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051843, 0.0051442

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 248
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 206

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 141

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013721, upper bound: 0.0013860
time: 1.92 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013721, upper bound: 0.0013851
time: 1.87 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0112433, 0.0111346
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051598, 0.0051718

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 248

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 187

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013599, upper bound: 0.0013727
time: 1.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013934, upper bound: 0.0013543
time: 1.75 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 4.43 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0013509, upper bound: 0.0013951
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0013611, upper bound: 0.0013694
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0013704, upper bound: 0.0013932
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0013652, upper bound: 0.0014053
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0013690, upper bound: 0.0013904
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0013588, upper bound: 0.0014050
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0013499, upper bound: 0.0013898
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0013704, upper bound: 0.0013626
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0013518, upper bound: 0.0013952
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0013599, upper bound: 0.0013647
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0013565, upper bound: 0.0013855
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0013668, upper bound: 0.0013612
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0013396, upper bound: 0.0013631
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0013396, upper bound: 0.0013631
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0013724, upper bound: 0.0013756
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0013715, upper bound: 0.0013906
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0012342, upper bound: 0.0012572
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0012342, upper bound: 0.0012572
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0013395, upper bound: 0.0013631
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0013395, upper bound: 0.0013631
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0013612, upper bound: 0.0013668
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0013855, upper bound: 0.0013565
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0012502, upper bound: 0.0012387
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0012502, upper bound: 0.0012387
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0013953, upper bound: 0.0013718
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0013840, upper bound: 0.0013770
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0013577, upper bound: 0.0013422
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0013577, upper bound: 0.0013422
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0010622, upper bound: 0.0010719
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0010622, upper bound: 0.0010719
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0013519, upper bound: 0.0013473
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0013519, upper bound: 0.0013473
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0013643, upper bound: 0.0013917
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0013592, upper bound: 0.0013986
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0012494, upper bound: 0.0012406
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0012494, upper bound: 0.0012406
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0013642, upper bound: 0.0013973
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0013642, upper bound: 0.0013969
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0013519, upper bound: 0.0013478
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0013519, upper bound: 0.0013478
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0013642, upper bound: 0.0013973
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0013642, upper bound: 0.0013970
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0013614, upper bound: 0.0013745
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0013774, upper bound: 0.0013578
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0013425, upper bound: 0.0013566
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0013425, upper bound: 0.0013567
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0013924, upper bound: 0.0013708
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0013924, upper bound: 0.0013704
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0013767, upper bound: 0.0013828
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0013652, upper bound: 0.0013954
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0013640, upper bound: 0.0013394
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0013640, upper bound: 0.0013394
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0013425, upper bound: 0.0013567
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0013425, upper bound: 0.0013568
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0013597, upper bound: 0.0013719
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0013935, upper bound: 0.0013543
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0013721, upper bound: 0.0013860
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0013721, upper bound: 0.0013851
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0013599, upper bound: 0.0013727
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 2, lower bound: -0.0013934, upper bound: 0.0013543

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0111117, 0.0116969
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0052222, 0.0051580

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 248
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 157

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 248

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0010468, upper bound: 0.0010494
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0010468, upper bound: 0.0010494
time: 0.80 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0112478, 0.0115098
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0052017, 0.0051729

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 248
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 157

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 187

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013364, upper bound: 0.0013625
time: 1.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0013544, upper bound: 0.0013455
time: 1.70 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0108877, 0.0114089
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051899, 0.0051327

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 248
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 65

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 142

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012322, upper bound: 0.0012520
time: 1.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012322, upper bound: 0.0012520
time: 1.34 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0108809, 0.0114181
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051909, 0.0051320

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 248
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 157

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 65

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013643, upper bound: 0.0013900
time: 2.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013580, upper bound: 0.0014046
time: 1.81 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0107651, 0.0111386
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051615, 0.0051205

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 248
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 142

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 141

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013634, upper bound: 0.0013832
time: 1.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013634, upper bound: 0.0013831
time: 1.87 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0106865, 0.0112139
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051698, 0.0051119

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 248

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 157

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013526, upper bound: 0.0013918
time: 2.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013508, upper bound: 0.0013982
time: 1.91 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0104878, 0.0109236
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051388, 0.0050909

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 248
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 141

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 206

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0013058, upper bound: 0.0013420
time: 1.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0013058, upper bound: 0.0013420
time: 1.63 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0106105, 0.0107889
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051240, 0.0051044

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 248

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 206

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0013253, upper bound: 0.0013199
time: 1.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0013253, upper bound: 0.0013199
time: 1.65 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0111134, 0.0117652
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0052294, 0.0051582

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 248

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 157

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013459, upper bound: 0.0013822
time: 2.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013435, upper bound: 0.0013887
time: 1.82 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0112308, 0.0115470
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0052056, 0.0051710

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 248
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 206

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0013242, upper bound: 0.0013276
time: 1.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0013242, upper bound: 0.0013276
time: 1.51 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0111882, 0.0116877
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0052209, 0.0051664

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 248
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 157

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013500, upper bound: 0.0013786
time: 2.11 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013444, upper bound: 0.0013800
time: 2.44 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0113094, 0.0114715
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051973, 0.0051797

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 248
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 141

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 248

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0010475, upper bound: 0.0010484
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0010475, upper bound: 0.0010484
time: 0.77 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0110980, 0.0110234
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051477, 0.0051559

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 248
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 65

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 111

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0013190, upper bound: 0.0013525
time: 1.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0013299, upper bound: 0.0013289
time: 1.55 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0108673, 0.0112345
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051709, 0.0051306

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 248
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 111

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 141

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0013220, upper bound: 0.0013485
time: 1.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0013222, upper bound: 0.0013485
time: 1.94 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0110510, 0.0111954
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051665, 0.0051507

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 248
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 65

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013716, upper bound: 0.0013604
time: 2.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013685, upper bound: 0.0013750
time: 1.97 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0110392, 0.0112306
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051705, 0.0051494

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 248

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 245

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013708, upper bound: 0.0013767
time: 2.07 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013648, upper bound: 0.0013895
time: 1.97 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0110743, 0.0110880
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051547, 0.0051533

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 248
type: DSZ, layer: 3, pos: 65

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 111

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0013206, upper bound: 0.0013525
time: 1.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0013298, upper bound: 0.0013253
time: 1.40 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0108510, 0.0112996
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051780, 0.0051288

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 248
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 111

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0013206, upper bound: 0.0013526
time: 1.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0013298, upper bound: 0.0013253
time: 1.45 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0114293, 0.0113094
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051797, 0.0051928

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 248
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 157

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 206

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0013254, upper bound: 0.0013314
time: 1.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0013254, upper bound: 0.0013314
time: 1.75 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0116183, 0.0111882
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051664, 0.0052136

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 248
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 65

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013847, upper bound: 0.0013492
time: 2.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0013500, upper bound: 0.0013556
time: 2.17 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0112687, 0.0109570
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051404, 0.0051746

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 248
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 157

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 142

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012500, upper bound: 0.0012368
time: 1.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012500, upper bound: 0.0012368
time: 1.77 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0112600, 0.0109633
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051410, 0.0051736

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 248
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 65

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 187

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013441, upper bound: 0.0013705
time: 1.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013767, upper bound: 0.0013488
time: 2.00 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0113178, 0.0108473
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051283, 0.0051800

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 248

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 141

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0013304, upper bound: 0.0013323
time: 1.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0013323, upper bound: 0.0013323
time: 1.69 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0110815, 0.0110589
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051515, 0.0051541

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 248

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0008549, upper bound: 0.0008554
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0008549, upper bound: 0.0008554
time: 0.72 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0109020, 0.0114005
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051890, 0.0051343

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 248
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 248

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 187

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013372, upper bound: 0.0013851
time: 2.00 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0013571, upper bound: 0.0013570
time: 1.94 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0108805, 0.0114357
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051929, 0.0051320

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 248
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 206

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 245

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013586, upper bound: 0.0013841
time: 2.09 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013533, upper bound: 0.0013978
time: 2.15 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0108891, 0.0113445
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051830, 0.0051330

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 248
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 111

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 245

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013637, upper bound: 0.0013781
time: 1.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013557, upper bound: 0.0013969
time: 1.78 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0109020, 0.0113661
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051853, 0.0051345

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 248
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 65

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 142

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 187

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013393, upper bound: 0.0013892
time: 1.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013570, upper bound: 0.0013593
time: 2.01 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0108789, 0.0114136
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051905, 0.0051319

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 248

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 187

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013392, upper bound: 0.0013892
time: 1.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013570, upper bound: 0.0013594
time: 2.15 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0108894, 0.0114352
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051928, 0.0051331

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 248
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 65

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013634, upper bound: 0.0013828
time: 2.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013543, upper bound: 0.0013962
time: 2.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0113182, 0.0115264
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0052031, 0.0051806

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 248
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 206

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 142

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012277, upper bound: 0.0012307
time: 1.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012277, upper bound: 0.0012307
time: 1.26 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0114827, 0.0113426
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051831, 0.0051987

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 248
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 157

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 245

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013769, upper bound: 0.0013572
time: 1.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013687, upper bound: 0.0013565
time: 1.99 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0112083, 0.0110383
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051494, 0.0051681

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 248
type: DSZ, layer: 3, pos: 65

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 245

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013919, upper bound: 0.0013610
time: 1.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013743, upper bound: 0.0013703
time: 2.08 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0112200, 0.0110599
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051517, 0.0051694

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 248
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 248

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 206

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0013502, upper bound: 0.0013211
time: 1.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0013502, upper bound: 0.0013213
time: 1.56 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0109702, 0.0112559
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051732, 0.0051418

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 248

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 65

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013759, upper bound: 0.0013788
time: 1.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013640, upper bound: 0.0013819
time: 1.85 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0109648, 0.0112651
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051742, 0.0051412

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 248
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 157

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 65

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013715, upper bound: 0.0013841
time: 1.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013580, upper bound: 0.0013947
time: 2.01 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0111903, 0.0108546
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051292, 0.0051660

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 248
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 141

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 142

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 187

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0013159, upper bound: 0.0013238
time: 1.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0013451, upper bound: 0.0013096
time: 1.53 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0110325, 0.0110656
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051523, 0.0051487

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 248
type: DSZ, layer: 3, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 142

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 141

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0013499, upper bound: 0.0013222
time: 1.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0013499, upper bound: 0.0013221
time: 1.57 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0107360, 0.0107208
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051164, 0.0051182

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 248
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 141

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0011838, upper bound: 0.0011722
time: 1.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0011838, upper bound: 0.0011722
time: 1.27 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0108853, 0.0106259
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051059, 0.0051346

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 248
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 142

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0011842, upper bound: 0.0011718
time: 1.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0011842, upper bound: 0.0011718
time: 1.27 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0109636, 0.0113362
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051820, 0.0051412

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 248

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 157

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0013189, upper bound: 0.0013428
time: 1.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0013191, upper bound: 0.0013428
time: 1.69 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0109698, 0.0113577
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051843, 0.0051419

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 248
type: DSZ, layer: 3, pos: 157

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 245

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013716, upper bound: 0.0013620
time: 1.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013657, upper bound: 0.0013847
time: 1.76 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0107401, 0.0107281
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051172, 0.0051186

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 248
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 65

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0011844, upper bound: 0.0011721
time: 0.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0011844, upper bound: 0.0011721
time: 0.93 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0108898, 0.0106316
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051066, 0.0051351

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 248

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 206

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0013451, upper bound: 0.0013096
time: 1.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0013451, upper bound: 0.0013096
time: 1.58 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 4.12 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0010468, upper bound: 0.0010494
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0010468, upper bound: 0.0010494
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0013364, upper bound: 0.0013625
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0013544, upper bound: 0.0013455
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0012322, upper bound: 0.0012520
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0012322, upper bound: 0.0012520
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0013643, upper bound: 0.0013900
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0013580, upper bound: 0.0014046
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0013634, upper bound: 0.0013832
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0013634, upper bound: 0.0013831
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0013526, upper bound: 0.0013918
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0013508, upper bound: 0.0013982
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0013058, upper bound: 0.0013420
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0013058, upper bound: 0.0013420
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0013253, upper bound: 0.0013199
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0013253, upper bound: 0.0013199
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0013459, upper bound: 0.0013822
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0013435, upper bound: 0.0013887
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0013242, upper bound: 0.0013276
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0013242, upper bound: 0.0013276
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0013500, upper bound: 0.0013786
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0013444, upper bound: 0.0013800
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0010475, upper bound: 0.0010484
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0010475, upper bound: 0.0010484
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0013190, upper bound: 0.0013525
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0013299, upper bound: 0.0013289
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0013220, upper bound: 0.0013485
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0013222, upper bound: 0.0013485
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0013716, upper bound: 0.0013604
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0013685, upper bound: 0.0013750
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0013708, upper bound: 0.0013767
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0013648, upper bound: 0.0013895
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0013206, upper bound: 0.0013525
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0013298, upper bound: 0.0013253
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0013206, upper bound: 0.0013526
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0013298, upper bound: 0.0013253
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0013254, upper bound: 0.0013314
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0013254, upper bound: 0.0013314
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0013847, upper bound: 0.0013492
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0013500, upper bound: 0.0013556
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0012500, upper bound: 0.0012368
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0012500, upper bound: 0.0012368
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0013441, upper bound: 0.0013705
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0013767, upper bound: 0.0013488
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0013304, upper bound: 0.0013323
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0013323, upper bound: 0.0013323
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0008549, upper bound: 0.0008554
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0008549, upper bound: 0.0008554
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0013372, upper bound: 0.0013851
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0013571, upper bound: 0.0013570
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0013586, upper bound: 0.0013841
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0013533, upper bound: 0.0013978
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0013637, upper bound: 0.0013781
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0013557, upper bound: 0.0013969
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0013393, upper bound: 0.0013892
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0013570, upper bound: 0.0013593
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0013392, upper bound: 0.0013892
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0013570, upper bound: 0.0013594
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0013634, upper bound: 0.0013828
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0013543, upper bound: 0.0013962
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0012277, upper bound: 0.0012307
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0012277, upper bound: 0.0012307
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0013769, upper bound: 0.0013572
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0013687, upper bound: 0.0013565
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0013919, upper bound: 0.0013610
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0013743, upper bound: 0.0013703
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0013502, upper bound: 0.0013211
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0013502, upper bound: 0.0013213
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0013759, upper bound: 0.0013788
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0013640, upper bound: 0.0013819
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0013715, upper bound: 0.0013841
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0013580, upper bound: 0.0013947
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0013159, upper bound: 0.0013238
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0013451, upper bound: 0.0013096
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0013499, upper bound: 0.0013222
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0013499, upper bound: 0.0013221
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0011838, upper bound: 0.0011722
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0011838, upper bound: 0.0011722
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0011842, upper bound: 0.0011718
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0011842, upper bound: 0.0011718
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0013189, upper bound: 0.0013428
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0013191, upper bound: 0.0013428
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0013716, upper bound: 0.0013620
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0013657, upper bound: 0.0013847
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0011844, upper bound: 0.0011721
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0011844, upper bound: 0.0011721
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0013451, upper bound: 0.0013096
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.12
Output dim: 2, lower bound: -0.0013451, upper bound: 0.0013096

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0104205, 0.0110150
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051488, 0.0050835

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 248
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 111

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 65

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0013356, upper bound: 0.0013565
time: 1.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013313, upper bound: 0.0013617
time: 1.80 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0107591, 0.0112104
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051694, 0.0051199

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 248
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 142

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 187

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013432, upper bound: 0.0013830
time: 2.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013569, upper bound: 0.0013637
time: 2.05 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0106827, 0.0112934
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051785, 0.0051115

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 248
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 157

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013518, upper bound: 0.0013905
time: 2.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013499, upper bound: 0.0013974
time: 2.18 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0108844, 0.0113480
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051834, 0.0051325

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 248
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 206

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 248

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 65

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012298, upper bound: 0.0012445
time: 1.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012298, upper bound: 0.0012445
time: 1.56 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0108960, 0.0113696
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051857, 0.0051338

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 157
type: DSZ, layer: 3, pos: 248
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 65

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 157

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0013180, upper bound: 0.0013284
time: 1.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0013180, upper bound: 0.0013284
time: 1.66 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0109047, 0.0113344
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051818, 0.0051346

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 248

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 65

### Candidate
type: DSZ, layer: 3, pos: 187

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013333, upper bound: 0.0013852
time: 1.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0013445, upper bound: 0.0013561
time: 2.14 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0108824, 0.0113696
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051857, 0.0051322

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 248
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 65

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 142

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 187

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013315, upper bound: 0.0013916
time: 2.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013428, upper bound: 0.0013577
time: 2.11 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0108944, 0.0114035
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051893, 0.0051335

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 248
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 141

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 142

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 248

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 65

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013452, upper bound: 0.0013736
time: 1.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013400, upper bound: 0.0013814
time: 1.95 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0108715, 0.0114387
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051932, 0.0051310

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 248
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 206

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 245

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013427, upper bound: 0.0013756
time: 2.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0013428, upper bound: 0.0013877
time: 2.15 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0042086, -0.0040461, -0.0042086, -0.0040461, -0.0001625, 0.0001625
1: -0.0102674, -0.0085546, -0.0102674, -0.0085546, -0.0017128, 0.0017128
2: 0.9641421, 0.9661976, 0.9641421, 0.9661976, -0.0020554, 0.0020554
3: -0.0181756, -0.0030149, -0.0181756, -0.0030149, -0.0109814, 0.0113260
4: -0.0004637, 0.0006893, -0.0004637, 0.0006893, -0.0011531, 0.0011531
5: 0.0168017, 0.0185467, 0.0168017, 0.0185467, -0.0017450, 0.0017450
6: 0.0016238, 0.0037345, 0.0016238, 0.0037345, -0.0021107, 0.0021107
7: -0.0069969, -0.0021511, -0.0069969, -0.0021511, -0.0048458, 0.0048458
8: 0.0111781, 0.0142952, 0.0111781, 0.0142952, -0.0031171, 0.0031171
9: 0.0178295, 0.0234360, 0.0178295, 0.0234360, -0.0051808, 0.0051431

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 141
type: DSZ, layer: 3, pos: 65
type: DSZ, layer: 3, pos: 245
type: DSZ, layer: 3, pos: 248
type: DSZ, layer: 3, pos: 142
type: DSZ, layer: 3, pos: 111
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 141

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0013053, upper bound: 0.0013334
time: 1.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0013053, upper bound: 0.0013334
time: 1.56 seconds

## Summary of splitting (split count: 8)
- Time for DS candidates: 3.94 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 3.94
Output dim: 2, lower bound: -0.0013356, upper bound: 0.0013565
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 3.94
Output dim: 2, lower bound: -0.0013313, upper bound: 0.0013617
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 3.94
Output dim: 2, lower bound: -0.0013432, upper bound: 0.0013830
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 3.94
Output dim: 2, lower bound: -0.0013569, upper bound: 0.0013637
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 3.94
Output dim: 2, lower bound: -0.0013518, upper bound: 0.0013905
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 3.94
Output dim: 2, lower bound: -0.0013499, upper bound: 0.0013974
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 3.94
Output dim: 2, lower bound: -0.0012298, upper bound: 0.0012445
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 3.94
Output dim: 2, lower bound: -0.0012298, upper bound: 0.0012445
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 3.94
Output dim: 2, lower bound: -0.0013180, upper bound: 0.0013284
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 3.94
Output dim: 2, lower bound: -0.0013180, upper bound: 0.0013284
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 3.94
Output dim: 2, lower bound: -0.0013333, upper bound: 0.0013852
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 3.94
Output dim: 2, lower bound: -0.0013445, upper bound: 0.0013561
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 3.94
Output dim: 2, lower bound: -0.0013315, upper bound: 0.0013916
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 3.94
Output dim: 2, lower bound: -0.0013428, upper bound: 0.0013577
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 3.94
Output dim: 2, lower bound: -0.0013452, upper bound: 0.0013736
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 3.94
Output dim: 2, lower bound: -0.0013400, upper bound: 0.0013814
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 3.94
Output dim: 2, lower bound: -0.0013427, upper bound: 0.0013756
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 3.94
Output dim: 2, lower bound: -0.0013428, upper bound: 0.0013877
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 3.94
Output dim: 2, lower bound: -0.0013053, upper bound: 0.0013334
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 3.94
Output dim: 2, lower bound: -0.0013053, upper bound: 0.0013334
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 2, lower bound: -0.0013444, upper bound: 0.0013800
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 2, lower bound: -0.0013716, upper bound: 0.0013604
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 2, lower bound: -0.0013685, upper bound: 0.0013750
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 2, lower bound: -0.0013708, upper bound: 0.0013767
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 2, lower bound: -0.0013648, upper bound: 0.0013895
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 2, lower bound: -0.0013847, upper bound: 0.0013492
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 2, lower bound: -0.0013441, upper bound: 0.0013705
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 2, lower bound: -0.0013767, upper bound: 0.0013488
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 2, lower bound: -0.0013372, upper bound: 0.0013851
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 2, lower bound: -0.0013586, upper bound: 0.0013841
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 2, lower bound: -0.0013533, upper bound: 0.0013978
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 2, lower bound: -0.0013637, upper bound: 0.0013781
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 2, lower bound: -0.0013557, upper bound: 0.0013969
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 2, lower bound: -0.0013393, upper bound: 0.0013892
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 2, lower bound: -0.0013570, upper bound: 0.0013593
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 2, lower bound: -0.0013392, upper bound: 0.0013892
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 2, lower bound: -0.0013570, upper bound: 0.0013594
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 2, lower bound: -0.0013634, upper bound: 0.0013828
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 2, lower bound: -0.0013543, upper bound: 0.0013962
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 2, lower bound: -0.0013769, upper bound: 0.0013572
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 2, lower bound: -0.0013687, upper bound: 0.0013565
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 2, lower bound: -0.0013919, upper bound: 0.0013610
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 2, lower bound: -0.0013743, upper bound: 0.0013703
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 2, lower bound: -0.0013759, upper bound: 0.0013788
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 2, lower bound: -0.0013640, upper bound: 0.0013819
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 2, lower bound: -0.0013715, upper bound: 0.0013841
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 2, lower bound: -0.0013580, upper bound: 0.0013947
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 2, lower bound: -0.0013716, upper bound: 0.0013620
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 2, lower bound: -0.0013657, upper bound: 0.0013847

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 3.39 + 598.46 = 601.85 seconds
