## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.037597364


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0949643, 0.0846384, -0.0949643, 0.0846384, -0.1796028, 0.1796028)
1: (-0.0607208, 0.0474269, -0.0607208, 0.0474269, -0.1081477, 0.1081477)
2: (-0.1246243, 0.0634956, -0.1246243, 0.0634956, -0.1881199, 0.1881199)
3: (0.9865127, 1.0434601, 0.9865127, 1.0434601, -0.0569474, 0.0569474)
4: (-0.0387707, 0.1049149, -0.0387707, 0.1049149, -0.1436856, 0.1436856)
5: (-0.0581584, 0.1370523, -0.0581584, 0.1370523, -0.1952107, 0.1952107)
6: (-0.1140255, 0.0998719, -0.1140255, 0.0998719, -0.2138974, 0.2138974)
7: (-0.0948062, 0.0074982, -0.0948062, 0.0074982, -0.1023044, 0.1023044)
8: (-0.0549271, 0.0979662, -0.0549271, 0.0979662, -0.1528933, 0.1528933)
9: (-0.0670059, 0.0810942, -0.0670059, 0.0810942, -0.1481002, 0.1481002)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.62 + 3.05 = 4.67 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.0408667, upper bound: 0.0408667

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 188

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0402928, upper bound: 0.0402928
time: 2.41 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0402928, upper bound: 0.0402928
time: 2.61 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 5.17 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 5.17
Output dim: 3, lower bound: -0.0402928, upper bound: 0.0402928
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 5.17
Output dim: 3, lower bound: -0.0402928, upper bound: 0.0402928

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.0949643, 0.0846384, -0.0949643, 0.0846384, -0.1796028, 0.1796028
1: -0.0607208, 0.0474269, -0.0607208, 0.0474269, -0.1081477, 0.1081477
2: -0.1246243, 0.0634956, -0.1246243, 0.0634956, -0.1881199, 0.1881199
3: 0.9865127, 1.0434601, 0.9865127, 1.0434601, -0.0569474, 0.0569474
4: -0.0387707, 0.1049149, -0.0387707, 0.1049149, -0.1436856, 0.1436856
5: -0.0581584, 0.1370523, -0.0581584, 0.1370523, -0.1952107, 0.1952107
6: -0.1140255, 0.0998719, -0.1140255, 0.0998719, -0.2138974, 0.2138974
7: -0.0948062, 0.0074982, -0.0948062, 0.0074982, -0.1023044, 0.1023044
8: -0.0549271, 0.0979662, -0.0549271, 0.0979662, -0.1528933, 0.1528933
9: -0.0670059, 0.0810942, -0.0670059, 0.0810942, -0.1481002, 0.1481002

Time for backsubstitution: 1.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0402347, upper bound: 0.0402456
time: 2.13 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0402456, upper bound: 0.0402347
time: 2.34 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.0949643, 0.0846384, -0.0949643, 0.0846384, -0.1796028, 0.1796028
1: -0.0607208, 0.0474269, -0.0607208, 0.0474269, -0.1081477, 0.1081477
2: -0.1246243, 0.0634956, -0.1246243, 0.0634956, -0.1881199, 0.1881199
3: 0.9865127, 1.0434601, 0.9865127, 1.0434601, -0.0569474, 0.0569474
4: -0.0387707, 0.1049149, -0.0387707, 0.1049149, -0.1436856, 0.1436856
5: -0.0581584, 0.1370523, -0.0581584, 0.1370523, -0.1952107, 0.1952107
6: -0.1140255, 0.0998719, -0.1140255, 0.0998719, -0.2138974, 0.2138974
7: -0.0948062, 0.0074982, -0.0948062, 0.0074982, -0.1023044, 0.1023044
8: -0.0549271, 0.0979662, -0.0549271, 0.0979662, -0.1528933, 0.1528933
9: -0.0670059, 0.0810942, -0.0670059, 0.0810942, -0.1481002, 0.1481002

Time for backsubstitution: 1.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0402347, upper bound: 0.0402456
time: 2.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0402347, upper bound: 0.0402347
time: 2.24 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 6.64 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 6.64
Output dim: 3, lower bound: -0.0402347, upper bound: 0.0402456
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 6.64
Output dim: 3, lower bound: -0.0402456, upper bound: 0.0402347
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 6.64
Output dim: 3, lower bound: -0.0402347, upper bound: 0.0402456
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 6.64
Output dim: 3, lower bound: -0.0402347, upper bound: 0.0402347

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0949643, 0.0846384, -0.0949643, 0.0846384, -0.1796028, 0.1796028
1: -0.0607208, 0.0474269, -0.0607208, 0.0474269, -0.1081477, 0.1081477
2: -0.1246243, 0.0634956, -0.1246243, 0.0634956, -0.1881199, 0.1881199
3: 0.9865127, 1.0434601, 0.9865127, 1.0434601, -0.0569474, 0.0569474
4: -0.0387707, 0.1049149, -0.0387707, 0.1049149, -0.1436856, 0.1436856
5: -0.0581584, 0.1370523, -0.0581584, 0.1370523, -0.1952107, 0.1952107
6: -0.1140255, 0.0998719, -0.1140255, 0.0998719, -0.2138974, 0.2138974
7: -0.0948062, 0.0074982, -0.0948062, 0.0074982, -0.1023044, 0.1023044
8: -0.0549271, 0.0979662, -0.0549271, 0.0979662, -0.1528933, 0.1528933
9: -0.0670059, 0.0810942, -0.0670059, 0.0810942, -0.1481002, 0.1481002

Time for backsubstitution: 1.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 207

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0396954, upper bound: 0.0397060
time: 2.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0396954, upper bound: 0.0397060
time: 2.57 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0949643, 0.0846384, -0.0949643, 0.0846384, -0.1796028, 0.1796028
1: -0.0607208, 0.0474269, -0.0607208, 0.0474269, -0.1081477, 0.1081477
2: -0.1246243, 0.0634956, -0.1246243, 0.0634956, -0.1881199, 0.1881199
3: 0.9865127, 1.0434601, 0.9865127, 1.0434601, -0.0569474, 0.0569474
4: -0.0387707, 0.1049149, -0.0387707, 0.1049149, -0.1436856, 0.1436856
5: -0.0581584, 0.1370523, -0.0581584, 0.1370523, -0.1952107, 0.1952107
6: -0.1140255, 0.0998719, -0.1140255, 0.0998719, -0.2138974, 0.2138974
7: -0.0948062, 0.0074982, -0.0948062, 0.0074982, -0.1023044, 0.1023044
8: -0.0549271, 0.0979662, -0.0549271, 0.0979662, -0.1528933, 0.1528933
9: -0.0670059, 0.0810942, -0.0670059, 0.0810942, -0.1481002, 0.1481002

Time for backsubstitution: 1.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 207

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0397060, upper bound: 0.0396954
time: 2.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0397060, upper bound: 0.0396954
time: 2.41 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0949643, 0.0846384, -0.0949643, 0.0846384, -0.1796028, 0.1796028
1: -0.0607208, 0.0474269, -0.0607208, 0.0474269, -0.1081477, 0.1081477
2: -0.1246243, 0.0634956, -0.1246243, 0.0634956, -0.1881199, 0.1881199
3: 0.9865127, 1.0434601, 0.9865127, 1.0434601, -0.0569474, 0.0569474
4: -0.0387707, 0.1049149, -0.0387707, 0.1049149, -0.1436856, 0.1436856
5: -0.0581584, 0.1370523, -0.0581584, 0.1370523, -0.1952107, 0.1952107
6: -0.1140255, 0.0998719, -0.1140255, 0.0998719, -0.2138974, 0.2138974
7: -0.0948062, 0.0074982, -0.0948062, 0.0074982, -0.1023044, 0.1023044
8: -0.0549271, 0.0979662, -0.0549271, 0.0979662, -0.1528933, 0.1528933
9: -0.0670059, 0.0810942, -0.0670059, 0.0810942, -0.1481002, 0.1481002

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 207

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0396954, upper bound: 0.0397060
time: 2.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0396954, upper bound: 0.0397060
time: 3.23 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0949643, 0.0846384, -0.0949643, 0.0846384, -0.1796028, 0.1796028
1: -0.0607208, 0.0474269, -0.0607208, 0.0474269, -0.1081477, 0.1081477
2: -0.1246243, 0.0634956, -0.1246243, 0.0634956, -0.1881199, 0.1881199
3: 0.9865127, 1.0434601, 0.9865127, 1.0434601, -0.0569474, 0.0569474
4: -0.0387707, 0.1049149, -0.0387707, 0.1049149, -0.1436856, 0.1436856
5: -0.0581584, 0.1370523, -0.0581584, 0.1370523, -0.1952107, 0.1952107
6: -0.1140255, 0.0998719, -0.1140255, 0.0998719, -0.2138974, 0.2138974
7: -0.0948062, 0.0074982, -0.0948062, 0.0074982, -0.1023044, 0.1023044
8: -0.0549271, 0.0979662, -0.0549271, 0.0979662, -0.1528933, 0.1528933
9: -0.0670059, 0.0810942, -0.0670059, 0.0810942, -0.1481002, 0.1481002

Time for backsubstitution: 1.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 207

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0397060, upper bound: 0.0396954
time: 2.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0397060, upper bound: 0.0396954
time: 2.11 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 5.80 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 5.80
Output dim: 3, lower bound: -0.0396954, upper bound: 0.0397060
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 5.80
Output dim: 3, lower bound: -0.0396954, upper bound: 0.0397060
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 5.80
Output dim: 3, lower bound: -0.0397060, upper bound: 0.0396954
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 5.80
Output dim: 3, lower bound: -0.0397060, upper bound: 0.0396954
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 5.80
Output dim: 3, lower bound: -0.0396954, upper bound: 0.0397060
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 5.80
Output dim: 3, lower bound: -0.0396954, upper bound: 0.0397060
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 5.80
Output dim: 3, lower bound: -0.0397060, upper bound: 0.0396954
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 5.80
Output dim: 3, lower bound: -0.0397060, upper bound: 0.0396954

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0949643, 0.0846384, -0.0949643, 0.0846384, -0.1796028, 0.1796028
1: -0.0607208, 0.0474269, -0.0607208, 0.0474269, -0.1081477, 0.1081477
2: -0.1246243, 0.0634956, -0.1246243, 0.0634956, -0.1881199, 0.1881199
3: 0.9865127, 1.0434601, 0.9865127, 1.0434601, -0.0569474, 0.0569474
4: -0.0387707, 0.1049149, -0.0387707, 0.1049149, -0.1436856, 0.1436856
5: -0.0581584, 0.1370523, -0.0581584, 0.1370523, -0.1952107, 0.1952107
6: -0.1140255, 0.0998719, -0.1140255, 0.0998719, -0.2138974, 0.2138974
7: -0.0948062, 0.0074982, -0.0948062, 0.0074982, -0.1023044, 0.1023044
8: -0.0549271, 0.0979662, -0.0549271, 0.0979662, -0.1528933, 0.1528933
9: -0.0670059, 0.0810942, -0.0670059, 0.0810942, -0.1481002, 0.1481002

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0396679, upper bound: 0.0396755
time: 2.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0396641, upper bound: 0.0396785
time: 2.77 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0949643, 0.0846384, -0.0949643, 0.0846384, -0.1796028, 0.1796028
1: -0.0607208, 0.0474269, -0.0607208, 0.0474269, -0.1081477, 0.1081477
2: -0.1246243, 0.0634956, -0.1246243, 0.0634956, -0.1881199, 0.1881199
3: 0.9865127, 1.0434601, 0.9865127, 1.0434601, -0.0569474, 0.0569474
4: -0.0387707, 0.1049149, -0.0387707, 0.1049149, -0.1436856, 0.1436856
5: -0.0581584, 0.1370523, -0.0581584, 0.1370523, -0.1952107, 0.1952107
6: -0.1140255, 0.0998719, -0.1140255, 0.0998719, -0.2138974, 0.2138974
7: -0.0948062, 0.0074982, -0.0948062, 0.0074982, -0.1023044, 0.1023044
8: -0.0549271, 0.0979662, -0.0549271, 0.0979662, -0.1528933, 0.1528933
9: -0.0670059, 0.0810942, -0.0670059, 0.0810942, -0.1481002, 0.1481002

Time for backsubstitution: 1.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0396679, upper bound: 0.0396755
time: 2.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0396641, upper bound: 0.0396786
time: 2.09 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0949643, 0.0846384, -0.0949643, 0.0846384, -0.1796028, 0.1796028
1: -0.0607208, 0.0474269, -0.0607208, 0.0474269, -0.1081477, 0.1081477
2: -0.1246243, 0.0634956, -0.1246243, 0.0634956, -0.1881199, 0.1881199
3: 0.9865127, 1.0434601, 0.9865127, 1.0434601, -0.0569474, 0.0569474
4: -0.0387707, 0.1049149, -0.0387707, 0.1049149, -0.1436856, 0.1436856
5: -0.0581584, 0.1370523, -0.0581584, 0.1370523, -0.1952107, 0.1952107
6: -0.1140255, 0.0998719, -0.1140255, 0.0998719, -0.2138974, 0.2138974
7: -0.0948062, 0.0074982, -0.0948062, 0.0074982, -0.1023044, 0.1023044
8: -0.0549271, 0.0979662, -0.0549271, 0.0979662, -0.1528933, 0.1528933
9: -0.0670059, 0.0810942, -0.0670059, 0.0810942, -0.1481002, 0.1481002

Time for backsubstitution: 1.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0396785, upper bound: 0.0396641
time: 2.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0396755, upper bound: 0.0396679
time: 2.12 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0949643, 0.0846384, -0.0949643, 0.0846384, -0.1796028, 0.1796028
1: -0.0607208, 0.0474269, -0.0607208, 0.0474269, -0.1081477, 0.1081477
2: -0.1246243, 0.0634956, -0.1246243, 0.0634956, -0.1881199, 0.1881199
3: 0.9865127, 1.0434601, 0.9865127, 1.0434601, -0.0569474, 0.0569474
4: -0.0387707, 0.1049149, -0.0387707, 0.1049149, -0.1436856, 0.1436856
5: -0.0581584, 0.1370523, -0.0581584, 0.1370523, -0.1952107, 0.1952107
6: -0.1140255, 0.0998719, -0.1140255, 0.0998719, -0.2138974, 0.2138974
7: -0.0948062, 0.0074982, -0.0948062, 0.0074982, -0.1023044, 0.1023044
8: -0.0549271, 0.0979662, -0.0549271, 0.0979662, -0.1528933, 0.1528933
9: -0.0670059, 0.0810942, -0.0670059, 0.0810942, -0.1481002, 0.1481002

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0396641, upper bound: 0.0396641
time: 2.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0396755, upper bound: 0.0396679
time: 2.10 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0949643, 0.0846384, -0.0949643, 0.0846384, -0.1796028, 0.1796028
1: -0.0607208, 0.0474269, -0.0607208, 0.0474269, -0.1081477, 0.1081477
2: -0.1246243, 0.0634956, -0.1246243, 0.0634956, -0.1881199, 0.1881199
3: 0.9865127, 1.0434601, 0.9865127, 1.0434601, -0.0569474, 0.0569474
4: -0.0387707, 0.1049149, -0.0387707, 0.1049149, -0.1436856, 0.1436856
5: -0.0581584, 0.1370523, -0.0581584, 0.1370523, -0.1952107, 0.1952107
6: -0.1140255, 0.0998719, -0.1140255, 0.0998719, -0.2138974, 0.2138974
7: -0.0948062, 0.0074982, -0.0948062, 0.0074982, -0.1023044, 0.1023044
8: -0.0549271, 0.0979662, -0.0549271, 0.0979662, -0.1528933, 0.1528933
9: -0.0670059, 0.0810942, -0.0670059, 0.0810942, -0.1481002, 0.1481002

Time for backsubstitution: 1.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0396641, upper bound: 0.0396755
time: 2.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0396641, upper bound: 0.0396786
time: 3.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0949643, 0.0846384, -0.0949643, 0.0846384, -0.1796028, 0.1796028
1: -0.0607208, 0.0474269, -0.0607208, 0.0474269, -0.1081477, 0.1081477
2: -0.1246243, 0.0634956, -0.1246243, 0.0634956, -0.1881199, 0.1881199
3: 0.9865127, 1.0434601, 0.9865127, 1.0434601, -0.0569474, 0.0569474
4: -0.0387707, 0.1049149, -0.0387707, 0.1049149, -0.1436856, 0.1436856
5: -0.0581584, 0.1370523, -0.0581584, 0.1370523, -0.1952107, 0.1952107
6: -0.1140255, 0.0998719, -0.1140255, 0.0998719, -0.2138974, 0.2138974
7: -0.0948062, 0.0074982, -0.0948062, 0.0074982, -0.1023044, 0.1023044
8: -0.0549271, 0.0979662, -0.0549271, 0.0979662, -0.1528933, 0.1528933
9: -0.0670059, 0.0810942, -0.0670059, 0.0810942, -0.1481002, 0.1481002

Time for backsubstitution: 1.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0396679, upper bound: 0.0396755
time: 2.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0396641, upper bound: 0.0396786
time: 2.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0949643, 0.0846384, -0.0949643, 0.0846384, -0.1796028, 0.1796028
1: -0.0607208, 0.0474269, -0.0607208, 0.0474269, -0.1081477, 0.1081477
2: -0.1246243, 0.0634956, -0.1246243, 0.0634956, -0.1881199, 0.1881199
3: 0.9865127, 1.0434601, 0.9865127, 1.0434601, -0.0569474, 0.0569474
4: -0.0387707, 0.1049149, -0.0387707, 0.1049149, -0.1436856, 0.1436856
5: -0.0581584, 0.1370523, -0.0581584, 0.1370523, -0.1952107, 0.1952107
6: -0.1140255, 0.0998719, -0.1140255, 0.0998719, -0.2138974, 0.2138974
7: -0.0948062, 0.0074982, -0.0948062, 0.0074982, -0.1023044, 0.1023044
8: -0.0549271, 0.0979662, -0.0549271, 0.0979662, -0.1528933, 0.1528933
9: -0.0670059, 0.0810942, -0.0670059, 0.0810942, -0.1481002, 0.1481002

Time for backsubstitution: 1.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0396785, upper bound: 0.0396641
time: 2.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0396755, upper bound: 0.0396679
time: 2.20 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0949643, 0.0846384, -0.0949643, 0.0846384, -0.1796028, 0.1796028
1: -0.0607208, 0.0474269, -0.0607208, 0.0474269, -0.1081477, 0.1081477
2: -0.1246243, 0.0634956, -0.1246243, 0.0634956, -0.1881199, 0.1881199
3: 0.9865127, 1.0434601, 0.9865127, 1.0434601, -0.0569474, 0.0569474
4: -0.0387707, 0.1049149, -0.0387707, 0.1049149, -0.1436856, 0.1436856
5: -0.0581584, 0.1370523, -0.0581584, 0.1370523, -0.1952107, 0.1952107
6: -0.1140255, 0.0998719, -0.1140255, 0.0998719, -0.2138974, 0.2138974
7: -0.0948062, 0.0074982, -0.0948062, 0.0074982, -0.1023044, 0.1023044
8: -0.0549271, 0.0979662, -0.0549271, 0.0979662, -0.1528933, 0.1528933
9: -0.0670059, 0.0810942, -0.0670059, 0.0810942, -0.1481002, 0.1481002

Time for backsubstitution: 1.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0396785, upper bound: 0.0396641
time: 1.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0396755, upper bound: 0.0396679
time: 2.19 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 5.72 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.72
Output dim: 3, lower bound: -0.0396679, upper bound: 0.0396755
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.72
Output dim: 3, lower bound: -0.0396641, upper bound: 0.0396785
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.72
Output dim: 3, lower bound: -0.0396679, upper bound: 0.0396755
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.72
Output dim: 3, lower bound: -0.0396641, upper bound: 0.0396786
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.72
Output dim: 3, lower bound: -0.0396785, upper bound: 0.0396641
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.72
Output dim: 3, lower bound: -0.0396755, upper bound: 0.0396679
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.72
Output dim: 3, lower bound: -0.0396641, upper bound: 0.0396641
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.72
Output dim: 3, lower bound: -0.0396755, upper bound: 0.0396679
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.72
Output dim: 3, lower bound: -0.0396641, upper bound: 0.0396755
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.72
Output dim: 3, lower bound: -0.0396641, upper bound: 0.0396786
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.72
Output dim: 3, lower bound: -0.0396679, upper bound: 0.0396755
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.72
Output dim: 3, lower bound: -0.0396641, upper bound: 0.0396786
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.72
Output dim: 3, lower bound: -0.0396785, upper bound: 0.0396641
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.72
Output dim: 3, lower bound: -0.0396755, upper bound: 0.0396679
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.72
Output dim: 3, lower bound: -0.0396785, upper bound: 0.0396641
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.72
Output dim: 3, lower bound: -0.0396755, upper bound: 0.0396679

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0949643, 0.0846384, -0.0949643, 0.0846384, -0.1796028, 0.1796028
1: -0.0607208, 0.0474269, -0.0607208, 0.0474269, -0.1081477, 0.1081477
2: -0.1246243, 0.0634956, -0.1246243, 0.0634956, -0.1881199, 0.1881199
3: 0.9865127, 1.0434601, 0.9865127, 1.0434601, -0.0569474, 0.0569474
4: -0.0387707, 0.1049149, -0.0387707, 0.1049149, -0.1436856, 0.1436856
5: -0.0581584, 0.1370523, -0.0581584, 0.1370523, -0.1952107, 0.1952107
6: -0.1140255, 0.0998719, -0.1140255, 0.0998719, -0.2138974, 0.2138974
7: -0.0948062, 0.0074982, -0.0948062, 0.0074982, -0.1023044, 0.1023044
8: -0.0549271, 0.0979662, -0.0549271, 0.0979662, -0.1528933, 0.1528933
9: -0.0670059, 0.0810942, -0.0670059, 0.0810942, -0.1481002, 0.1481002

Time for backsubstitution: 1.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 255

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393482, upper bound: 0.0393607
time: 2.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393521, upper bound: 0.0393604
time: 2.24 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0949643, 0.0846384, -0.0949643, 0.0846384, -0.1796028, 0.1796028
1: -0.0607208, 0.0474269, -0.0607208, 0.0474269, -0.1081477, 0.1081477
2: -0.1246243, 0.0634956, -0.1246243, 0.0634956, -0.1881199, 0.1881199
3: 0.9865127, 1.0434601, 0.9865127, 1.0434601, -0.0569474, 0.0569474
4: -0.0387707, 0.1049149, -0.0387707, 0.1049149, -0.1436856, 0.1436856
5: -0.0581584, 0.1370523, -0.0581584, 0.1370523, -0.1952107, 0.1952107
6: -0.1140255, 0.0998719, -0.1140255, 0.0998719, -0.2138974, 0.2138974
7: -0.0948062, 0.0074982, -0.0948062, 0.0074982, -0.1023044, 0.1023044
8: -0.0549271, 0.0979662, -0.0549271, 0.0979662, -0.1528933, 0.1528933
9: -0.0670059, 0.0810942, -0.0670059, 0.0810942, -0.1481002, 0.1481002

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 255

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393482, upper bound: 0.0393638
time: 2.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393482, upper bound: 0.0393636
time: 2.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0949643, 0.0846384, -0.0949643, 0.0846384, -0.1796028, 0.1796028
1: -0.0607208, 0.0474269, -0.0607208, 0.0474269, -0.1081477, 0.1081477
2: -0.1246243, 0.0634956, -0.1246243, 0.0634956, -0.1881199, 0.1881199
3: 0.9865127, 1.0434601, 0.9865127, 1.0434601, -0.0569474, 0.0569474
4: -0.0387707, 0.1049149, -0.0387707, 0.1049149, -0.1436856, 0.1436856
5: -0.0581584, 0.1370523, -0.0581584, 0.1370523, -0.1952107, 0.1952107
6: -0.1140255, 0.0998719, -0.1140255, 0.0998719, -0.2138974, 0.2138974
7: -0.0948062, 0.0074982, -0.0948062, 0.0074982, -0.1023044, 0.1023044
8: -0.0549271, 0.0979662, -0.0549271, 0.0979662, -0.1528933, 0.1528933
9: -0.0670059, 0.0810942, -0.0670059, 0.0810942, -0.1481002, 0.1481002

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 255

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393482, upper bound: 0.0393607
time: 3.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393521, upper bound: 0.0393604
time: 2.40 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0949643, 0.0846384, -0.0949643, 0.0846384, -0.1796028, 0.1796028
1: -0.0607208, 0.0474269, -0.0607208, 0.0474269, -0.1081477, 0.1081477
2: -0.1246243, 0.0634956, -0.1246243, 0.0634956, -0.1881199, 0.1881199
3: 0.9865127, 1.0434601, 0.9865127, 1.0434601, -0.0569474, 0.0569474
4: -0.0387707, 0.1049149, -0.0387707, 0.1049149, -0.1436856, 0.1436856
5: -0.0581584, 0.1370523, -0.0581584, 0.1370523, -0.1952107, 0.1952107
6: -0.1140255, 0.0998719, -0.1140255, 0.0998719, -0.2138974, 0.2138974
7: -0.0948062, 0.0074982, -0.0948062, 0.0074982, -0.1023044, 0.1023044
8: -0.0549271, 0.0979662, -0.0549271, 0.0979662, -0.1528933, 0.1528933
9: -0.0670059, 0.0810942, -0.0670059, 0.0810942, -0.1481002, 0.1481002

Time for backsubstitution: 1.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 255

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393482, upper bound: 0.0393638
time: 2.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393482, upper bound: 0.0393636
time: 2.28 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0949643, 0.0846384, -0.0949643, 0.0846384, -0.1796028, 0.1796028
1: -0.0607208, 0.0474269, -0.0607208, 0.0474269, -0.1081477, 0.1081477
2: -0.1246243, 0.0634956, -0.1246243, 0.0634956, -0.1881199, 0.1881199
3: 0.9865127, 1.0434601, 0.9865127, 1.0434601, -0.0569474, 0.0569474
4: -0.0387707, 0.1049149, -0.0387707, 0.1049149, -0.1436856, 0.1436856
5: -0.0581584, 0.1370523, -0.0581584, 0.1370523, -0.1952107, 0.1952107
6: -0.1140255, 0.0998719, -0.1140255, 0.0998719, -0.2138974, 0.2138974
7: -0.0948062, 0.0074982, -0.0948062, 0.0074982, -0.1023044, 0.1023044
8: -0.0549271, 0.0979662, -0.0549271, 0.0979662, -0.1528933, 0.1528933
9: -0.0670059, 0.0810942, -0.0670059, 0.0810942, -0.1481002, 0.1481002

Time for backsubstitution: 1.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 255

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393636, upper bound: 0.0393484
time: 1.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393638, upper bound: 0.0393482
time: 2.08 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0949643, 0.0846384, -0.0949643, 0.0846384, -0.1796028, 0.1796028
1: -0.0607208, 0.0474269, -0.0607208, 0.0474269, -0.1081477, 0.1081477
2: -0.1246243, 0.0634956, -0.1246243, 0.0634956, -0.1881199, 0.1881199
3: 0.9865127, 1.0434601, 0.9865127, 1.0434601, -0.0569474, 0.0569474
4: -0.0387707, 0.1049149, -0.0387707, 0.1049149, -0.1436856, 0.1436856
5: -0.0581584, 0.1370523, -0.0581584, 0.1370523, -0.1952107, 0.1952107
6: -0.1140255, 0.0998719, -0.1140255, 0.0998719, -0.2138974, 0.2138974
7: -0.0948062, 0.0074982, -0.0948062, 0.0074982, -0.1023044, 0.1023044
8: -0.0549271, 0.0979662, -0.0549271, 0.0979662, -0.1528933, 0.1528933
9: -0.0670059, 0.0810942, -0.0670059, 0.0810942, -0.1481002, 0.1481002

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 255

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393604, upper bound: 0.0393521
time: 2.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393482, upper bound: 0.0393521
time: 2.87 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0949643, 0.0846384, -0.0949643, 0.0846384, -0.1796028, 0.1796028
1: -0.0607208, 0.0474269, -0.0607208, 0.0474269, -0.1081477, 0.1081477
2: -0.1246243, 0.0634956, -0.1246243, 0.0634956, -0.1881199, 0.1881199
3: 0.9865127, 1.0434601, 0.9865127, 1.0434601, -0.0569474, 0.0569474
4: -0.0387707, 0.1049149, -0.0387707, 0.1049149, -0.1436856, 0.1436856
5: -0.0581584, 0.1370523, -0.0581584, 0.1370523, -0.1952107, 0.1952107
6: -0.1140255, 0.0998719, -0.1140255, 0.0998719, -0.2138974, 0.2138974
7: -0.0948062, 0.0074982, -0.0948062, 0.0074982, -0.1023044, 0.1023044
8: -0.0549271, 0.0979662, -0.0549271, 0.0979662, -0.1528933, 0.1528933
9: -0.0670059, 0.0810942, -0.0670059, 0.0810942, -0.1481002, 0.1481002

Time for backsubstitution: 1.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 255

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393636, upper bound: 0.0393484
time: 2.09 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393638, upper bound: 0.0393482
time: 2.12 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0949643, 0.0846384, -0.0949643, 0.0846384, -0.1796028, 0.1796028
1: -0.0607208, 0.0474269, -0.0607208, 0.0474269, -0.1081477, 0.1081477
2: -0.1246243, 0.0634956, -0.1246243, 0.0634956, -0.1881199, 0.1881199
3: 0.9865127, 1.0434601, 0.9865127, 1.0434601, -0.0569474, 0.0569474
4: -0.0387707, 0.1049149, -0.0387707, 0.1049149, -0.1436856, 0.1436856
5: -0.0581584, 0.1370523, -0.0581584, 0.1370523, -0.1952107, 0.1952107
6: -0.1140255, 0.0998719, -0.1140255, 0.0998719, -0.2138974, 0.2138974
7: -0.0948062, 0.0074982, -0.0948062, 0.0074982, -0.1023044, 0.1023044
8: -0.0549271, 0.0979662, -0.0549271, 0.0979662, -0.1528933, 0.1528933
9: -0.0670059, 0.0810942, -0.0670059, 0.0810942, -0.1481002, 0.1481002

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 255

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393604, upper bound: 0.0393521
time: 2.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393482, upper bound: 0.0393521
time: 2.48 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0949643, 0.0846384, -0.0949643, 0.0846384, -0.1796028, 0.1796028
1: -0.0607208, 0.0474269, -0.0607208, 0.0474269, -0.1081477, 0.1081477
2: -0.1246243, 0.0634956, -0.1246243, 0.0634956, -0.1881199, 0.1881199
3: 0.9865127, 1.0434601, 0.9865127, 1.0434601, -0.0569474, 0.0569474
4: -0.0387707, 0.1049149, -0.0387707, 0.1049149, -0.1436856, 0.1436856
5: -0.0581584, 0.1370523, -0.0581584, 0.1370523, -0.1952107, 0.1952107
6: -0.1140255, 0.0998719, -0.1140255, 0.0998719, -0.2138974, 0.2138974
7: -0.0948062, 0.0074982, -0.0948062, 0.0074982, -0.1023044, 0.1023044
8: -0.0549271, 0.0979662, -0.0549271, 0.0979662, -0.1528933, 0.1528933
9: -0.0670059, 0.0810942, -0.0670059, 0.0810942, -0.1481002, 0.1481002

Time for backsubstitution: 1.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 255

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393521, upper bound: 0.0393607
time: 2.08 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393521, upper bound: 0.0393604
time: 2.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0949643, 0.0846384, -0.0949643, 0.0846384, -0.1796028, 0.1796028
1: -0.0607208, 0.0474269, -0.0607208, 0.0474269, -0.1081477, 0.1081477
2: -0.1246243, 0.0634956, -0.1246243, 0.0634956, -0.1881199, 0.1881199
3: 0.9865127, 1.0434601, 0.9865127, 1.0434601, -0.0569474, 0.0569474
4: -0.0387707, 0.1049149, -0.0387707, 0.1049149, -0.1436856, 0.1436856
5: -0.0581584, 0.1370523, -0.0581584, 0.1370523, -0.1952107, 0.1952107
6: -0.1140255, 0.0998719, -0.1140255, 0.0998719, -0.2138974, 0.2138974
7: -0.0948062, 0.0074982, -0.0948062, 0.0074982, -0.1023044, 0.1023044
8: -0.0549271, 0.0979662, -0.0549271, 0.0979662, -0.1528933, 0.1528933
9: -0.0670059, 0.0810942, -0.0670059, 0.0810942, -0.1481002, 0.1481002

Time for backsubstitution: 1.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 255

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393482, upper bound: 0.0393638
time: 2.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393482, upper bound: 0.0393636
time: 2.15 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0949643, 0.0846384, -0.0949643, 0.0846384, -0.1796028, 0.1796028
1: -0.0607208, 0.0474269, -0.0607208, 0.0474269, -0.1081477, 0.1081477
2: -0.1246243, 0.0634956, -0.1246243, 0.0634956, -0.1881199, 0.1881199
3: 0.9865127, 1.0434601, 0.9865127, 1.0434601, -0.0569474, 0.0569474
4: -0.0387707, 0.1049149, -0.0387707, 0.1049149, -0.1436856, 0.1436856
5: -0.0581584, 0.1370523, -0.0581584, 0.1370523, -0.1952107, 0.1952107
6: -0.1140255, 0.0998719, -0.1140255, 0.0998719, -0.2138974, 0.2138974
7: -0.0948062, 0.0074982, -0.0948062, 0.0074982, -0.1023044, 0.1023044
8: -0.0549271, 0.0979662, -0.0549271, 0.0979662, -0.1528933, 0.1528933
9: -0.0670059, 0.0810942, -0.0670059, 0.0810942, -0.1481002, 0.1481002

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 255

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393521, upper bound: 0.0393607
time: 2.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393521, upper bound: 0.0393604
time: 2.29 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0949643, 0.0846384, -0.0949643, 0.0846384, -0.1796028, 0.1796028
1: -0.0607208, 0.0474269, -0.0607208, 0.0474269, -0.1081477, 0.1081477
2: -0.1246243, 0.0634956, -0.1246243, 0.0634956, -0.1881199, 0.1881199
3: 0.9865127, 1.0434601, 0.9865127, 1.0434601, -0.0569474, 0.0569474
4: -0.0387707, 0.1049149, -0.0387707, 0.1049149, -0.1436856, 0.1436856
5: -0.0581584, 0.1370523, -0.0581584, 0.1370523, -0.1952107, 0.1952107
6: -0.1140255, 0.0998719, -0.1140255, 0.0998719, -0.2138974, 0.2138974
7: -0.0948062, 0.0074982, -0.0948062, 0.0074982, -0.1023044, 0.1023044
8: -0.0549271, 0.0979662, -0.0549271, 0.0979662, -0.1528933, 0.1528933
9: -0.0670059, 0.0810942, -0.0670059, 0.0810942, -0.1481002, 0.1481002

Time for backsubstitution: 1.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 255

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393482, upper bound: 0.0393638
time: 2.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393482, upper bound: 0.0393636
time: 3.20 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0949643, 0.0846384, -0.0949643, 0.0846384, -0.1796028, 0.1796028
1: -0.0607208, 0.0474269, -0.0607208, 0.0474269, -0.1081477, 0.1081477
2: -0.1246243, 0.0634956, -0.1246243, 0.0634956, -0.1881199, 0.1881199
3: 0.9865127, 1.0434601, 0.9865127, 1.0434601, -0.0569474, 0.0569474
4: -0.0387707, 0.1049149, -0.0387707, 0.1049149, -0.1436856, 0.1436856
5: -0.0581584, 0.1370523, -0.0581584, 0.1370523, -0.1952107, 0.1952107
6: -0.1140255, 0.0998719, -0.1140255, 0.0998719, -0.2138974, 0.2138974
7: -0.0948062, 0.0074982, -0.0948062, 0.0074982, -0.1023044, 0.1023044
8: -0.0549271, 0.0979662, -0.0549271, 0.0979662, -0.1528933, 0.1528933
9: -0.0670059, 0.0810942, -0.0670059, 0.0810942, -0.1481002, 0.1481002

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 255

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393636, upper bound: 0.0393484
time: 2.13 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393638, upper bound: 0.0393482
time: 2.14 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0949643, 0.0846384, -0.0949643, 0.0846384, -0.1796028, 0.1796028
1: -0.0607208, 0.0474269, -0.0607208, 0.0474269, -0.1081477, 0.1081477
2: -0.1246243, 0.0634956, -0.1246243, 0.0634956, -0.1881199, 0.1881199
3: 0.9865127, 1.0434601, 0.9865127, 1.0434601, -0.0569474, 0.0569474
4: -0.0387707, 0.1049149, -0.0387707, 0.1049149, -0.1436856, 0.1436856
5: -0.0581584, 0.1370523, -0.0581584, 0.1370523, -0.1952107, 0.1952107
6: -0.1140255, 0.0998719, -0.1140255, 0.0998719, -0.2138974, 0.2138974
7: -0.0948062, 0.0074982, -0.0948062, 0.0074982, -0.1023044, 0.1023044
8: -0.0549271, 0.0979662, -0.0549271, 0.0979662, -0.1528933, 0.1528933
9: -0.0670059, 0.0810942, -0.0670059, 0.0810942, -0.1481002, 0.1481002

Time for backsubstitution: 1.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 255

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393604, upper bound: 0.0393521
time: 2.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393482, upper bound: 0.0393521
time: 3.47 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0949643, 0.0846384, -0.0949643, 0.0846384, -0.1796028, 0.1796028
1: -0.0607208, 0.0474269, -0.0607208, 0.0474269, -0.1081477, 0.1081477
2: -0.1246243, 0.0634956, -0.1246243, 0.0634956, -0.1881199, 0.1881199
3: 0.9865127, 1.0434601, 0.9865127, 1.0434601, -0.0569474, 0.0569474
4: -0.0387707, 0.1049149, -0.0387707, 0.1049149, -0.1436856, 0.1436856
5: -0.0581584, 0.1370523, -0.0581584, 0.1370523, -0.1952107, 0.1952107
6: -0.1140255, 0.0998719, -0.1140255, 0.0998719, -0.2138974, 0.2138974
7: -0.0948062, 0.0074982, -0.0948062, 0.0074982, -0.1023044, 0.1023044
8: -0.0549271, 0.0979662, -0.0549271, 0.0979662, -0.1528933, 0.1528933
9: -0.0670059, 0.0810942, -0.0670059, 0.0810942, -0.1481002, 0.1481002

Time for backsubstitution: 1.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 255

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393636, upper bound: 0.0393484
time: 2.13 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393638, upper bound: 0.0393482
time: 2.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0949643, 0.0846384, -0.0949643, 0.0846384, -0.1796028, 0.1796028
1: -0.0607208, 0.0474269, -0.0607208, 0.0474269, -0.1081477, 0.1081477
2: -0.1246243, 0.0634956, -0.1246243, 0.0634956, -0.1881199, 0.1881199
3: 0.9865127, 1.0434601, 0.9865127, 1.0434601, -0.0569474, 0.0569474
4: -0.0387707, 0.1049149, -0.0387707, 0.1049149, -0.1436856, 0.1436856
5: -0.0581584, 0.1370523, -0.0581584, 0.1370523, -0.1952107, 0.1952107
6: -0.1140255, 0.0998719, -0.1140255, 0.0998719, -0.2138974, 0.2138974
7: -0.0948062, 0.0074982, -0.0948062, 0.0074982, -0.1023044, 0.1023044
8: -0.0549271, 0.0979662, -0.0549271, 0.0979662, -0.1528933, 0.1528933
9: -0.0670059, 0.0810942, -0.0670059, 0.0810942, -0.1481002, 0.1481002

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 255

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393604, upper bound: 0.0393521
time: 2.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393482, upper bound: 0.0393521
time: 2.58 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 10.46 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.46
Output dim: 3, lower bound: -0.0393482, upper bound: 0.0393607
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.46
Output dim: 3, lower bound: -0.0393521, upper bound: 0.0393604
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.46
Output dim: 3, lower bound: -0.0393482, upper bound: 0.0393638
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.46
Output dim: 3, lower bound: -0.0393482, upper bound: 0.0393636
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.46
Output dim: 3, lower bound: -0.0393482, upper bound: 0.0393607
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.46
Output dim: 3, lower bound: -0.0393521, upper bound: 0.0393604
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.46
Output dim: 3, lower bound: -0.0393482, upper bound: 0.0393638
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.46
Output dim: 3, lower bound: -0.0393482, upper bound: 0.0393636
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.46
Output dim: 3, lower bound: -0.0393636, upper bound: 0.0393484
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.46
Output dim: 3, lower bound: -0.0393638, upper bound: 0.0393482
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.46
Output dim: 3, lower bound: -0.0393604, upper bound: 0.0393521
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.46
Output dim: 3, lower bound: -0.0393482, upper bound: 0.0393521
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.46
Output dim: 3, lower bound: -0.0393636, upper bound: 0.0393484
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.46
Output dim: 3, lower bound: -0.0393638, upper bound: 0.0393482
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.46
Output dim: 3, lower bound: -0.0393604, upper bound: 0.0393521
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.46
Output dim: 3, lower bound: -0.0393482, upper bound: 0.0393521
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.46
Output dim: 3, lower bound: -0.0393521, upper bound: 0.0393607
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.46
Output dim: 3, lower bound: -0.0393521, upper bound: 0.0393604
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.46
Output dim: 3, lower bound: -0.0393482, upper bound: 0.0393638
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.46
Output dim: 3, lower bound: -0.0393482, upper bound: 0.0393636
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.46
Output dim: 3, lower bound: -0.0393521, upper bound: 0.0393607
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.46
Output dim: 3, lower bound: -0.0393521, upper bound: 0.0393604
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.46
Output dim: 3, lower bound: -0.0393482, upper bound: 0.0393638
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.46
Output dim: 3, lower bound: -0.0393482, upper bound: 0.0393636
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.46
Output dim: 3, lower bound: -0.0393636, upper bound: 0.0393484
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.46
Output dim: 3, lower bound: -0.0393638, upper bound: 0.0393482
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.46
Output dim: 3, lower bound: -0.0393604, upper bound: 0.0393521
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.46
Output dim: 3, lower bound: -0.0393482, upper bound: 0.0393521
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.46
Output dim: 3, lower bound: -0.0393636, upper bound: 0.0393484
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.46
Output dim: 3, lower bound: -0.0393638, upper bound: 0.0393482
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.46
Output dim: 3, lower bound: -0.0393604, upper bound: 0.0393521
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.46
Output dim: 3, lower bound: -0.0393482, upper bound: 0.0393521

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0949643, 0.0846384, -0.0949643, 0.0846384, -0.1796028, 0.1796028
1: -0.0607208, 0.0474269, -0.0607208, 0.0474269, -0.1081477, 0.1081477
2: -0.1246243, 0.0634956, -0.1246243, 0.0634956, -0.1881199, 0.1881199
3: 0.9865127, 1.0434601, 0.9865127, 1.0434601, -0.0569474, 0.0569474
4: -0.0387707, 0.1049149, -0.0387707, 0.1049149, -0.1436856, 0.1436856
5: -0.0581584, 0.1370523, -0.0581584, 0.1370523, -0.1952107, 0.1952107
6: -0.1140255, 0.0998719, -0.1140255, 0.0998719, -0.2138974, 0.2138974
7: -0.0948062, 0.0074982, -0.0948062, 0.0074982, -0.1023044, 0.1023044
8: -0.0549271, 0.0979662, -0.0549271, 0.0979662, -0.1528933, 0.1528933
9: -0.0670059, 0.0810942, -0.0670059, 0.0810942, -0.1481002, 0.1481002

Time for backsubstitution: 1.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Candidate
type: DSZ, layer: 1, pos: 213

### Candidate
type: DSZ, layer: 1, pos: 118

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392845, upper bound: 0.0392988
time: 6.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392902, upper bound: 0.0392901
time: 2.27 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0949643, 0.0846384, -0.0949643, 0.0846384, -0.1796028, 0.1796028
1: -0.0607208, 0.0474269, -0.0607208, 0.0474269, -0.1081477, 0.1081477
2: -0.1246243, 0.0634956, -0.1246243, 0.0634956, -0.1881199, 0.1881199
3: 0.9865127, 1.0434601, 0.9865127, 1.0434601, -0.0569474, 0.0569474
4: -0.0387707, 0.1049149, -0.0387707, 0.1049149, -0.1436856, 0.1436856
5: -0.0581584, 0.1370523, -0.0581584, 0.1370523, -0.1952107, 0.1952107
6: -0.1140255, 0.0998719, -0.1140255, 0.0998719, -0.2138974, 0.2138974
7: -0.0948062, 0.0074982, -0.0948062, 0.0074982, -0.1023044, 0.1023044
8: -0.0549271, 0.0979662, -0.0549271, 0.0979662, -0.1528933, 0.1528933
9: -0.0670059, 0.0810942, -0.0670059, 0.0810942, -0.1481002, 0.1481002

Time for backsubstitution: 1.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Candidate
type: DSZ, layer: 1, pos: 213

### Candidate
type: DSZ, layer: 1, pos: 118

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392845, upper bound: 0.0392985
time: 2.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392903, upper bound: 0.0392898
time: 2.03 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0949643, 0.0846384, -0.0949643, 0.0846384, -0.1796028, 0.1796028
1: -0.0607208, 0.0474269, -0.0607208, 0.0474269, -0.1081477, 0.1081477
2: -0.1246243, 0.0634956, -0.1246243, 0.0634956, -0.1881199, 0.1881199
3: 0.9865127, 1.0434601, 0.9865127, 1.0434601, -0.0569474, 0.0569474
4: -0.0387707, 0.1049149, -0.0387707, 0.1049149, -0.1436856, 0.1436856
5: -0.0581584, 0.1370523, -0.0581584, 0.1370523, -0.1952107, 0.1952107
6: -0.1140255, 0.0998719, -0.1140255, 0.0998719, -0.2138974, 0.2138974
7: -0.0948062, 0.0074982, -0.0948062, 0.0074982, -0.1023044, 0.1023044
8: -0.0549271, 0.0979662, -0.0549271, 0.0979662, -0.1528933, 0.1528933
9: -0.0670059, 0.0810942, -0.0670059, 0.0810942, -0.1481002, 0.1481002

Time for backsubstitution: 1.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Candidate
type: DSZ, layer: 1, pos: 213

### Candidate
type: DSZ, layer: 1, pos: 118

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392810, upper bound: 0.0393020
time: 23.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392864, upper bound: 0.0392932
time: 2.17 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0949643, 0.0846384, -0.0949643, 0.0846384, -0.1796028, 0.1796028
1: -0.0607208, 0.0474269, -0.0607208, 0.0474269, -0.1081477, 0.1081477
2: -0.1246243, 0.0634956, -0.1246243, 0.0634956, -0.1881199, 0.1881199
3: 0.9865127, 1.0434601, 0.9865127, 1.0434601, -0.0569474, 0.0569474
4: -0.0387707, 0.1049149, -0.0387707, 0.1049149, -0.1436856, 0.1436856
5: -0.0581584, 0.1370523, -0.0581584, 0.1370523, -0.1952107, 0.1952107
6: -0.1140255, 0.0998719, -0.1140255, 0.0998719, -0.2138974, 0.2138974
7: -0.0948062, 0.0074982, -0.0948062, 0.0074982, -0.1023044, 0.1023044
8: -0.0549271, 0.0979662, -0.0549271, 0.0979662, -0.1528933, 0.1528933
9: -0.0670059, 0.0810942, -0.0670059, 0.0810942, -0.1481002, 0.1481002

Time for backsubstitution: 1.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Candidate
type: DSZ, layer: 1, pos: 213

### Candidate
type: DSZ, layer: 1, pos: 118

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392810, upper bound: 0.0393018
time: 2.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392810, upper bound: 0.0392932
time: 2.46 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0949643, 0.0846384, -0.0949643, 0.0846384, -0.1796028, 0.1796028
1: -0.0607208, 0.0474269, -0.0607208, 0.0474269, -0.1081477, 0.1081477
2: -0.1246243, 0.0634956, -0.1246243, 0.0634956, -0.1881199, 0.1881199
3: 0.9865127, 1.0434601, 0.9865127, 1.0434601, -0.0569474, 0.0569474
4: -0.0387707, 0.1049149, -0.0387707, 0.1049149, -0.1436856, 0.1436856
5: -0.0581584, 0.1370523, -0.0581584, 0.1370523, -0.1952107, 0.1952107
6: -0.1140255, 0.0998719, -0.1140255, 0.0998719, -0.2138974, 0.2138974
7: -0.0948062, 0.0074982, -0.0948062, 0.0074982, -0.1023044, 0.1023044
8: -0.0549271, 0.0979662, -0.0549271, 0.0979662, -0.1528933, 0.1528933
9: -0.0670059, 0.0810942, -0.0670059, 0.0810942, -0.1481002, 0.1481002

Time for backsubstitution: 1.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Candidate
type: DSZ, layer: 1, pos: 213

### Candidate
type: DSZ, layer: 1, pos: 118

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392810, upper bound: 0.0392988
time: 2.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392902, upper bound: 0.0392900
time: 2.33 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0949643, 0.0846384, -0.0949643, 0.0846384, -0.1796028, 0.1796028
1: -0.0607208, 0.0474269, -0.0607208, 0.0474269, -0.1081477, 0.1081477
2: -0.1246243, 0.0634956, -0.1246243, 0.0634956, -0.1881199, 0.1881199
3: 0.9865127, 1.0434601, 0.9865127, 1.0434601, -0.0569474, 0.0569474
4: -0.0387707, 0.1049149, -0.0387707, 0.1049149, -0.1436856, 0.1436856
5: -0.0581584, 0.1370523, -0.0581584, 0.1370523, -0.1952107, 0.1952107
6: -0.1140255, 0.0998719, -0.1140255, 0.0998719, -0.2138974, 0.2138974
7: -0.0948062, 0.0074982, -0.0948062, 0.0074982, -0.1023044, 0.1023044
8: -0.0549271, 0.0979662, -0.0549271, 0.0979662, -0.1528933, 0.1528933
9: -0.0670059, 0.0810942, -0.0670059, 0.0810942, -0.1481002, 0.1481002

Time for backsubstitution: 1.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Candidate
type: DSZ, layer: 1, pos: 213

### Candidate
type: DSZ, layer: 1, pos: 118

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392845, upper bound: 0.0392985
time: 2.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392903, upper bound: 0.0392898
time: 1.95 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0949643, 0.0846384, -0.0949643, 0.0846384, -0.1796028, 0.1796028
1: -0.0607208, 0.0474269, -0.0607208, 0.0474269, -0.1081477, 0.1081477
2: -0.1246243, 0.0634956, -0.1246243, 0.0634956, -0.1881199, 0.1881199
3: 0.9865127, 1.0434601, 0.9865127, 1.0434601, -0.0569474, 0.0569474
4: -0.0387707, 0.1049149, -0.0387707, 0.1049149, -0.1436856, 0.1436856
5: -0.0581584, 0.1370523, -0.0581584, 0.1370523, -0.1952107, 0.1952107
6: -0.1140255, 0.0998719, -0.1140255, 0.0998719, -0.2138974, 0.2138974
7: -0.0948062, 0.0074982, -0.0948062, 0.0074982, -0.1023044, 0.1023044
8: -0.0549271, 0.0979662, -0.0549271, 0.0979662, -0.1528933, 0.1528933
9: -0.0670059, 0.0810942, -0.0670059, 0.0810942, -0.1481002, 0.1481002

Time for backsubstitution: 1.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Candidate
type: DSZ, layer: 1, pos: 213

### Candidate
type: DSZ, layer: 1, pos: 118

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392810, upper bound: 0.0393020
time: 1.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392810, upper bound: 0.0392932
time: 2.35 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0949643, 0.0846384, -0.0949643, 0.0846384, -0.1796028, 0.1796028
1: -0.0607208, 0.0474269, -0.0607208, 0.0474269, -0.1081477, 0.1081477
2: -0.1246243, 0.0634956, -0.1246243, 0.0634956, -0.1881199, 0.1881199
3: 0.9865127, 1.0434601, 0.9865127, 1.0434601, -0.0569474, 0.0569474
4: -0.0387707, 0.1049149, -0.0387707, 0.1049149, -0.1436856, 0.1436856
5: -0.0581584, 0.1370523, -0.0581584, 0.1370523, -0.1952107, 0.1952107
6: -0.1140255, 0.0998719, -0.1140255, 0.0998719, -0.2138974, 0.2138974
7: -0.0948062, 0.0074982, -0.0948062, 0.0074982, -0.1023044, 0.1023044
8: -0.0549271, 0.0979662, -0.0549271, 0.0979662, -0.1528933, 0.1528933
9: -0.0670059, 0.0810942, -0.0670059, 0.0810942, -0.1481002, 0.1481002

Time for backsubstitution: 1.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Candidate
type: DSZ, layer: 1, pos: 213

### Candidate
type: DSZ, layer: 1, pos: 118

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392810, upper bound: 0.0393018
time: 2.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392810, upper bound: 0.0392932
time: 2.43 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0949643, 0.0846384, -0.0949643, 0.0846384, -0.1796028, 0.1796028
1: -0.0607208, 0.0474269, -0.0607208, 0.0474269, -0.1081477, 0.1081477
2: -0.1246243, 0.0634956, -0.1246243, 0.0634956, -0.1881199, 0.1881199
3: 0.9865127, 1.0434601, 0.9865127, 1.0434601, -0.0569474, 0.0569474
4: -0.0387707, 0.1049149, -0.0387707, 0.1049149, -0.1436856, 0.1436856
5: -0.0581584, 0.1370523, -0.0581584, 0.1370523, -0.1952107, 0.1952107
6: -0.1140255, 0.0998719, -0.1140255, 0.0998719, -0.2138974, 0.2138974
7: -0.0948062, 0.0074982, -0.0948062, 0.0074982, -0.1023044, 0.1023044
8: -0.0549271, 0.0979662, -0.0549271, 0.0979662, -0.1528933, 0.1528933
9: -0.0670059, 0.0810942, -0.0670059, 0.0810942, -0.1481002, 0.1481002

Time for backsubstitution: 1.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Candidate
type: DSZ, layer: 1, pos: 213

### Candidate
type: DSZ, layer: 1, pos: 118

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392845, upper bound: 0.0392866
time: 2.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393018, upper bound: 0.0392811
time: 2.61 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0949643, 0.0846384, -0.0949643, 0.0846384, -0.1796028, 0.1796028
1: -0.0607208, 0.0474269, -0.0607208, 0.0474269, -0.1081477, 0.1081477
2: -0.1246243, 0.0634956, -0.1246243, 0.0634956, -0.1881199, 0.1881199
3: 0.9865127, 1.0434601, 0.9865127, 1.0434601, -0.0569474, 0.0569474
4: -0.0387707, 0.1049149, -0.0387707, 0.1049149, -0.1436856, 0.1436856
5: -0.0581584, 0.1370523, -0.0581584, 0.1370523, -0.1952107, 0.1952107
6: -0.1140255, 0.0998719, -0.1140255, 0.0998719, -0.2138974, 0.2138974
7: -0.0948062, 0.0074982, -0.0948062, 0.0074982, -0.1023044, 0.1023044
8: -0.0549271, 0.0979662, -0.0549271, 0.0979662, -0.1528933, 0.1528933
9: -0.0670059, 0.0810942, -0.0670059, 0.0810942, -0.1481002, 0.1481002

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Candidate
type: DSZ, layer: 1, pos: 213

### Candidate
type: DSZ, layer: 1, pos: 118

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392932, upper bound: 0.0392864
time: 2.36 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393020, upper bound: 0.0392810
time: 2.17 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0949643, 0.0846384, -0.0949643, 0.0846384, -0.1796028, 0.1796028
1: -0.0607208, 0.0474269, -0.0607208, 0.0474269, -0.1081477, 0.1081477
2: -0.1246243, 0.0634956, -0.1246243, 0.0634956, -0.1881199, 0.1881199
3: 0.9865127, 1.0434601, 0.9865127, 1.0434601, -0.0569474, 0.0569474
4: -0.0387707, 0.1049149, -0.0387707, 0.1049149, -0.1436856, 0.1436856
5: -0.0581584, 0.1370523, -0.0581584, 0.1370523, -0.1952107, 0.1952107
6: -0.1140255, 0.0998719, -0.1140255, 0.0998719, -0.2138974, 0.2138974
7: -0.0948062, 0.0074982, -0.0948062, 0.0074982, -0.1023044, 0.1023044
8: -0.0549271, 0.0979662, -0.0549271, 0.0979662, -0.1528933, 0.1528933
9: -0.0670059, 0.0810942, -0.0670059, 0.0810942, -0.1481002, 0.1481002

Time for backsubstitution: 1.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Candidate
type: DSZ, layer: 1, pos: 213

### Candidate
type: DSZ, layer: 1, pos: 118

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392898, upper bound: 0.0392902
time: 2.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392985, upper bound: 0.0392845
time: 2.32 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0949643, 0.0846384, -0.0949643, 0.0846384, -0.1796028, 0.1796028
1: -0.0607208, 0.0474269, -0.0607208, 0.0474269, -0.1081477, 0.1081477
2: -0.1246243, 0.0634956, -0.1246243, 0.0634956, -0.1881199, 0.1881199
3: 0.9865127, 1.0434601, 0.9865127, 1.0434601, -0.0569474, 0.0569474
4: -0.0387707, 0.1049149, -0.0387707, 0.1049149, -0.1436856, 0.1436856
5: -0.0581584, 0.1370523, -0.0581584, 0.1370523, -0.1952107, 0.1952107
6: -0.1140255, 0.0998719, -0.1140255, 0.0998719, -0.2138974, 0.2138974
7: -0.0948062, 0.0074982, -0.0948062, 0.0074982, -0.1023044, 0.1023044
8: -0.0549271, 0.0979662, -0.0549271, 0.0979662, -0.1528933, 0.1528933
9: -0.0670059, 0.0810942, -0.0670059, 0.0810942, -0.1481002, 0.1481002

Time for backsubstitution: 1.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Candidate
type: DSZ, layer: 1, pos: 213

### Candidate
type: DSZ, layer: 1, pos: 118

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392900, upper bound: 0.0392902
time: 2.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392989, upper bound: 0.0392845
time: 2.33 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0949643, 0.0846384, -0.0949643, 0.0846384, -0.1796028, 0.1796028
1: -0.0607208, 0.0474269, -0.0607208, 0.0474269, -0.1081477, 0.1081477
2: -0.1246243, 0.0634956, -0.1246243, 0.0634956, -0.1881199, 0.1881199
3: 0.9865127, 1.0434601, 0.9865127, 1.0434601, -0.0569474, 0.0569474
4: -0.0387707, 0.1049149, -0.0387707, 0.1049149, -0.1436856, 0.1436856
5: -0.0581584, 0.1370523, -0.0581584, 0.1370523, -0.1952107, 0.1952107
6: -0.1140255, 0.0998719, -0.1140255, 0.0998719, -0.2138974, 0.2138974
7: -0.0948062, 0.0074982, -0.0948062, 0.0074982, -0.1023044, 0.1023044
8: -0.0549271, 0.0979662, -0.0549271, 0.0979662, -0.1528933, 0.1528933
9: -0.0670059, 0.0810942, -0.0670059, 0.0810942, -0.1481002, 0.1481002

Time for backsubstitution: 1.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Candidate
type: DSZ, layer: 1, pos: 213

### Candidate
type: DSZ, layer: 1, pos: 118

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392810, upper bound: 0.0392866
time: 2.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393018, upper bound: 0.0392811
time: 2.39 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0949643, 0.0846384, -0.0949643, 0.0846384, -0.1796028, 0.1796028
1: -0.0607208, 0.0474269, -0.0607208, 0.0474269, -0.1081477, 0.1081477
2: -0.1246243, 0.0634956, -0.1246243, 0.0634956, -0.1881199, 0.1881199
3: 0.9865127, 1.0434601, 0.9865127, 1.0434601, -0.0569474, 0.0569474
4: -0.0387707, 0.1049149, -0.0387707, 0.1049149, -0.1436856, 0.1436856
5: -0.0581584, 0.1370523, -0.0581584, 0.1370523, -0.1952107, 0.1952107
6: -0.1140255, 0.0998719, -0.1140255, 0.0998719, -0.2138974, 0.2138974
7: -0.0948062, 0.0074982, -0.0948062, 0.0074982, -0.1023044, 0.1023044
8: -0.0549271, 0.0979662, -0.0549271, 0.0979662, -0.1528933, 0.1528933
9: -0.0670059, 0.0810942, -0.0670059, 0.0810942, -0.1481002, 0.1481002

Time for backsubstitution: 1.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Candidate
type: DSZ, layer: 1, pos: 213

### Candidate
type: DSZ, layer: 1, pos: 118

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392932, upper bound: 0.0392864
time: 2.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393020, upper bound: 0.0392810
time: 2.76 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0949643, 0.0846384, -0.0949643, 0.0846384, -0.1796028, 0.1796028
1: -0.0607208, 0.0474269, -0.0607208, 0.0474269, -0.1081477, 0.1081477
2: -0.1246243, 0.0634956, -0.1246243, 0.0634956, -0.1881199, 0.1881199
3: 0.9865127, 1.0434601, 0.9865127, 1.0434601, -0.0569474, 0.0569474
4: -0.0387707, 0.1049149, -0.0387707, 0.1049149, -0.1436856, 0.1436856
5: -0.0581584, 0.1370523, -0.0581584, 0.1370523, -0.1952107, 0.1952107
6: -0.1140255, 0.0998719, -0.1140255, 0.0998719, -0.2138974, 0.2138974
7: -0.0948062, 0.0074982, -0.0948062, 0.0074982, -0.1023044, 0.1023044
8: -0.0549271, 0.0979662, -0.0549271, 0.0979662, -0.1528933, 0.1528933
9: -0.0670059, 0.0810942, -0.0670059, 0.0810942, -0.1481002, 0.1481002

Time for backsubstitution: 1.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Candidate
type: DSZ, layer: 1, pos: 213

### Candidate
type: DSZ, layer: 1, pos: 118

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392898, upper bound: 0.0392903
time: 2.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392985, upper bound: 0.0392845
time: 2.36 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0949643, 0.0846384, -0.0949643, 0.0846384, -0.1796028, 0.1796028
1: -0.0607208, 0.0474269, -0.0607208, 0.0474269, -0.1081477, 0.1081477
2: -0.1246243, 0.0634956, -0.1246243, 0.0634956, -0.1881199, 0.1881199
3: 0.9865127, 1.0434601, 0.9865127, 1.0434601, -0.0569474, 0.0569474
4: -0.0387707, 0.1049149, -0.0387707, 0.1049149, -0.1436856, 0.1436856
5: -0.0581584, 0.1370523, -0.0581584, 0.1370523, -0.1952107, 0.1952107
6: -0.1140255, 0.0998719, -0.1140255, 0.0998719, -0.2138974, 0.2138974
7: -0.0948062, 0.0074982, -0.0948062, 0.0074982, -0.1023044, 0.1023044
8: -0.0549271, 0.0979662, -0.0549271, 0.0979662, -0.1528933, 0.1528933
9: -0.0670059, 0.0810942, -0.0670059, 0.0810942, -0.1481002, 0.1481002

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Candidate
type: DSZ, layer: 1, pos: 213

### Candidate
type: DSZ, layer: 1, pos: 118

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392900, upper bound: 0.0392902
time: 2.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392989, upper bound: 0.0392845
time: 2.46 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0949643, 0.0846384, -0.0949643, 0.0846384, -0.1796028, 0.1796028
1: -0.0607208, 0.0474269, -0.0607208, 0.0474269, -0.1081477, 0.1081477
2: -0.1246243, 0.0634956, -0.1246243, 0.0634956, -0.1881199, 0.1881199
3: 0.9865127, 1.0434601, 0.9865127, 1.0434601, -0.0569474, 0.0569474
4: -0.0387707, 0.1049149, -0.0387707, 0.1049149, -0.1436856, 0.1436856
5: -0.0581584, 0.1370523, -0.0581584, 0.1370523, -0.1952107, 0.1952107
6: -0.1140255, 0.0998719, -0.1140255, 0.0998719, -0.2138974, 0.2138974
7: -0.0948062, 0.0074982, -0.0948062, 0.0074982, -0.1023044, 0.1023044
8: -0.0549271, 0.0979662, -0.0549271, 0.0979662, -0.1528933, 0.1528933
9: -0.0670059, 0.0810942, -0.0670059, 0.0810942, -0.1481002, 0.1481002

Time for backsubstitution: 1.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Candidate
type: DSZ, layer: 1, pos: 213

### Candidate
type: DSZ, layer: 1, pos: 118

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392845, upper bound: 0.0392988
time: 2.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392902, upper bound: 0.0392901
time: 2.18 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0949643, 0.0846384, -0.0949643, 0.0846384, -0.1796028, 0.1796028
1: -0.0607208, 0.0474269, -0.0607208, 0.0474269, -0.1081477, 0.1081477
2: -0.1246243, 0.0634956, -0.1246243, 0.0634956, -0.1881199, 0.1881199
3: 0.9865127, 1.0434601, 0.9865127, 1.0434601, -0.0569474, 0.0569474
4: -0.0387707, 0.1049149, -0.0387707, 0.1049149, -0.1436856, 0.1436856
5: -0.0581584, 0.1370523, -0.0581584, 0.1370523, -0.1952107, 0.1952107
6: -0.1140255, 0.0998719, -0.1140255, 0.0998719, -0.2138974, 0.2138974
7: -0.0948062, 0.0074982, -0.0948062, 0.0074982, -0.1023044, 0.1023044
8: -0.0549271, 0.0979662, -0.0549271, 0.0979662, -0.1528933, 0.1528933
9: -0.0670059, 0.0810942, -0.0670059, 0.0810942, -0.1481002, 0.1481002

Time for backsubstitution: 1.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Candidate
type: DSZ, layer: 1, pos: 213

### Candidate
type: DSZ, layer: 1, pos: 118

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392845, upper bound: 0.0392985
time: 2.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392903, upper bound: 0.0392898
time: 1.87 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0949643, 0.0846384, -0.0949643, 0.0846384, -0.1796028, 0.1796028
1: -0.0607208, 0.0474269, -0.0607208, 0.0474269, -0.1081477, 0.1081477
2: -0.1246243, 0.0634956, -0.1246243, 0.0634956, -0.1881199, 0.1881199
3: 0.9865127, 1.0434601, 0.9865127, 1.0434601, -0.0569474, 0.0569474
4: -0.0387707, 0.1049149, -0.0387707, 0.1049149, -0.1436856, 0.1436856
5: -0.0581584, 0.1370523, -0.0581584, 0.1370523, -0.1952107, 0.1952107
6: -0.1140255, 0.0998719, -0.1140255, 0.0998719, -0.2138974, 0.2138974
7: -0.0948062, 0.0074982, -0.0948062, 0.0074982, -0.1023044, 0.1023044
8: -0.0549271, 0.0979662, -0.0549271, 0.0979662, -0.1528933, 0.1528933
9: -0.0670059, 0.0810942, -0.0670059, 0.0810942, -0.1481002, 0.1481002

Time for backsubstitution: 1.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Candidate
type: DSZ, layer: 1, pos: 213

### Candidate
type: DSZ, layer: 1, pos: 118

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392810, upper bound: 0.0393020
time: 2.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392864, upper bound: 0.0392933
time: 3.47 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0949643, 0.0846384, -0.0949643, 0.0846384, -0.1796028, 0.1796028
1: -0.0607208, 0.0474269, -0.0607208, 0.0474269, -0.1081477, 0.1081477
2: -0.1246243, 0.0634956, -0.1246243, 0.0634956, -0.1881199, 0.1881199
3: 0.9865127, 1.0434601, 0.9865127, 1.0434601, -0.0569474, 0.0569474
4: -0.0387707, 0.1049149, -0.0387707, 0.1049149, -0.1436856, 0.1436856
5: -0.0581584, 0.1370523, -0.0581584, 0.1370523, -0.1952107, 0.1952107
6: -0.1140255, 0.0998719, -0.1140255, 0.0998719, -0.2138974, 0.2138974
7: -0.0948062, 0.0074982, -0.0948062, 0.0074982, -0.1023044, 0.1023044
8: -0.0549271, 0.0979662, -0.0549271, 0.0979662, -0.1528933, 0.1528933
9: -0.0670059, 0.0810942, -0.0670059, 0.0810942, -0.1481002, 0.1481002

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Candidate
type: DSZ, layer: 1, pos: 213

### Candidate
type: DSZ, layer: 1, pos: 118

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392810, upper bound: 0.0393018
time: 2.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392810, upper bound: 0.0392932
time: 2.34 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0949643, 0.0846384, -0.0949643, 0.0846384, -0.1796028, 0.1796028
1: -0.0607208, 0.0474269, -0.0607208, 0.0474269, -0.1081477, 0.1081477
2: -0.1246243, 0.0634956, -0.1246243, 0.0634956, -0.1881199, 0.1881199
3: 0.9865127, 1.0434601, 0.9865127, 1.0434601, -0.0569474, 0.0569474
4: -0.0387707, 0.1049149, -0.0387707, 0.1049149, -0.1436856, 0.1436856
5: -0.0581584, 0.1370523, -0.0581584, 0.1370523, -0.1952107, 0.1952107
6: -0.1140255, 0.0998719, -0.1140255, 0.0998719, -0.2138974, 0.2138974
7: -0.0948062, 0.0074982, -0.0948062, 0.0074982, -0.1023044, 0.1023044
8: -0.0549271, 0.0979662, -0.0549271, 0.0979662, -0.1528933, 0.1528933
9: -0.0670059, 0.0810942, -0.0670059, 0.0810942, -0.1481002, 0.1481002

Time for backsubstitution: 1.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Candidate
type: DSZ, layer: 1, pos: 213

### Candidate
type: DSZ, layer: 1, pos: 118

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392845, upper bound: 0.0392989
time: 2.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392902, upper bound: 0.0392900
time: 2.07 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0949643, 0.0846384, -0.0949643, 0.0846384, -0.1796028, 0.1796028
1: -0.0607208, 0.0474269, -0.0607208, 0.0474269, -0.1081477, 0.1081477
2: -0.1246243, 0.0634956, -0.1246243, 0.0634956, -0.1881199, 0.1881199
3: 0.9865127, 1.0434601, 0.9865127, 1.0434601, -0.0569474, 0.0569474
4: -0.0387707, 0.1049149, -0.0387707, 0.1049149, -0.1436856, 0.1436856
5: -0.0581584, 0.1370523, -0.0581584, 0.1370523, -0.1952107, 0.1952107
6: -0.1140255, 0.0998719, -0.1140255, 0.0998719, -0.2138974, 0.2138974
7: -0.0948062, 0.0074982, -0.0948062, 0.0074982, -0.1023044, 0.1023044
8: -0.0549271, 0.0979662, -0.0549271, 0.0979662, -0.1528933, 0.1528933
9: -0.0670059, 0.0810942, -0.0670059, 0.0810942, -0.1481002, 0.1481002

Time for backsubstitution: 1.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Candidate
type: DSZ, layer: 1, pos: 213

### Candidate
type: DSZ, layer: 1, pos: 118

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392845, upper bound: 0.0392985
time: 2.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392903, upper bound: 0.0392898
time: 1.91 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0949643, 0.0846384, -0.0949643, 0.0846384, -0.1796028, 0.1796028
1: -0.0607208, 0.0474269, -0.0607208, 0.0474269, -0.1081477, 0.1081477
2: -0.1246243, 0.0634956, -0.1246243, 0.0634956, -0.1881199, 0.1881199
3: 0.9865127, 1.0434601, 0.9865127, 1.0434601, -0.0569474, 0.0569474
4: -0.0387707, 0.1049149, -0.0387707, 0.1049149, -0.1436856, 0.1436856
5: -0.0581584, 0.1370523, -0.0581584, 0.1370523, -0.1952107, 0.1952107
6: -0.1140255, 0.0998719, -0.1140255, 0.0998719, -0.2138974, 0.2138974
7: -0.0948062, 0.0074982, -0.0948062, 0.0074982, -0.1023044, 0.1023044
8: -0.0549271, 0.0979662, -0.0549271, 0.0979662, -0.1528933, 0.1528933
9: -0.0670059, 0.0810942, -0.0670059, 0.0810942, -0.1481002, 0.1481002

Time for backsubstitution: 1.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Candidate
type: DSZ, layer: 1, pos: 213

### Candidate
type: DSZ, layer: 1, pos: 118

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392810, upper bound: 0.0393020
time: 2.09 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392864, upper bound: 0.0392932
time: 3.28 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0949643, 0.0846384, -0.0949643, 0.0846384, -0.1796028, 0.1796028
1: -0.0607208, 0.0474269, -0.0607208, 0.0474269, -0.1081477, 0.1081477
2: -0.1246243, 0.0634956, -0.1246243, 0.0634956, -0.1881199, 0.1881199
3: 0.9865127, 1.0434601, 0.9865127, 1.0434601, -0.0569474, 0.0569474
4: -0.0387707, 0.1049149, -0.0387707, 0.1049149, -0.1436856, 0.1436856
5: -0.0581584, 0.1370523, -0.0581584, 0.1370523, -0.1952107, 0.1952107
6: -0.1140255, 0.0998719, -0.1140255, 0.0998719, -0.2138974, 0.2138974
7: -0.0948062, 0.0074982, -0.0948062, 0.0074982, -0.1023044, 0.1023044
8: -0.0549271, 0.0979662, -0.0549271, 0.0979662, -0.1528933, 0.1528933
9: -0.0670059, 0.0810942, -0.0670059, 0.0810942, -0.1481002, 0.1481002

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Candidate
type: DSZ, layer: 1, pos: 213

### Candidate
type: DSZ, layer: 1, pos: 118

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392810, upper bound: 0.0393018
time: 2.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392810, upper bound: 0.0392932
time: 2.13 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0949643, 0.0846384, -0.0949643, 0.0846384, -0.1796028, 0.1796028
1: -0.0607208, 0.0474269, -0.0607208, 0.0474269, -0.1081477, 0.1081477
2: -0.1246243, 0.0634956, -0.1246243, 0.0634956, -0.1881199, 0.1881199
3: 0.9865127, 1.0434601, 0.9865127, 1.0434601, -0.0569474, 0.0569474
4: -0.0387707, 0.1049149, -0.0387707, 0.1049149, -0.1436856, 0.1436856
5: -0.0581584, 0.1370523, -0.0581584, 0.1370523, -0.1952107, 0.1952107
6: -0.1140255, 0.0998719, -0.1140255, 0.0998719, -0.2138974, 0.2138974
7: -0.0948062, 0.0074982, -0.0948062, 0.0074982, -0.1023044, 0.1023044
8: -0.0549271, 0.0979662, -0.0549271, 0.0979662, -0.1528933, 0.1528933
9: -0.0670059, 0.0810942, -0.0670059, 0.0810942, -0.1481002, 0.1481002

Time for backsubstitution: 1.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Candidate
type: DSZ, layer: 1, pos: 213

### Candidate
type: DSZ, layer: 1, pos: 118

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392810, upper bound: 0.0392866
time: 2.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393018, upper bound: 0.0392810
time: 2.34 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0949643, 0.0846384, -0.0949643, 0.0846384, -0.1796028, 0.1796028
1: -0.0607208, 0.0474269, -0.0607208, 0.0474269, -0.1081477, 0.1081477
2: -0.1246243, 0.0634956, -0.1246243, 0.0634956, -0.1881199, 0.1881199
3: 0.9865127, 1.0434601, 0.9865127, 1.0434601, -0.0569474, 0.0569474
4: -0.0387707, 0.1049149, -0.0387707, 0.1049149, -0.1436856, 0.1436856
5: -0.0581584, 0.1370523, -0.0581584, 0.1370523, -0.1952107, 0.1952107
6: -0.1140255, 0.0998719, -0.1140255, 0.0998719, -0.2138974, 0.2138974
7: -0.0948062, 0.0074982, -0.0948062, 0.0074982, -0.1023044, 0.1023044
8: -0.0549271, 0.0979662, -0.0549271, 0.0979662, -0.1528933, 0.1528933
9: -0.0670059, 0.0810942, -0.0670059, 0.0810942, -0.1481002, 0.1481002

Time for backsubstitution: 1.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Candidate
type: DSZ, layer: 1, pos: 213

### Candidate
type: DSZ, layer: 1, pos: 118

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392845, upper bound: 0.0392864
time: 2.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393020, upper bound: 0.0392810
time: 2.20 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0949643, 0.0846384, -0.0949643, 0.0846384, -0.1796028, 0.1796028
1: -0.0607208, 0.0474269, -0.0607208, 0.0474269, -0.1081477, 0.1081477
2: -0.1246243, 0.0634956, -0.1246243, 0.0634956, -0.1881199, 0.1881199
3: 0.9865127, 1.0434601, 0.9865127, 1.0434601, -0.0569474, 0.0569474
4: -0.0387707, 0.1049149, -0.0387707, 0.1049149, -0.1436856, 0.1436856
5: -0.0581584, 0.1370523, -0.0581584, 0.1370523, -0.1952107, 0.1952107
6: -0.1140255, 0.0998719, -0.1140255, 0.0998719, -0.2138974, 0.2138974
7: -0.0948062, 0.0074982, -0.0948062, 0.0074982, -0.1023044, 0.1023044
8: -0.0549271, 0.0979662, -0.0549271, 0.0979662, -0.1528933, 0.1528933
9: -0.0670059, 0.0810942, -0.0670059, 0.0810942, -0.1481002, 0.1481002

Time for backsubstitution: 1.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Candidate
type: DSZ, layer: 1, pos: 213

### Candidate
type: DSZ, layer: 1, pos: 118

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392898, upper bound: 0.0392903
time: 2.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392985, upper bound: 0.0392846
time: 2.29 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0949643, 0.0846384, -0.0949643, 0.0846384, -0.1796028, 0.1796028
1: -0.0607208, 0.0474269, -0.0607208, 0.0474269, -0.1081477, 0.1081477
2: -0.1246243, 0.0634956, -0.1246243, 0.0634956, -0.1881199, 0.1881199
3: 0.9865127, 1.0434601, 0.9865127, 1.0434601, -0.0569474, 0.0569474
4: -0.0387707, 0.1049149, -0.0387707, 0.1049149, -0.1436856, 0.1436856
5: -0.0581584, 0.1370523, -0.0581584, 0.1370523, -0.1952107, 0.1952107
6: -0.1140255, 0.0998719, -0.1140255, 0.0998719, -0.2138974, 0.2138974
7: -0.0948062, 0.0074982, -0.0948062, 0.0074982, -0.1023044, 0.1023044
8: -0.0549271, 0.0979662, -0.0549271, 0.0979662, -0.1528933, 0.1528933
9: -0.0670059, 0.0810942, -0.0670059, 0.0810942, -0.1481002, 0.1481002

Time for backsubstitution: 1.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Candidate
type: DSZ, layer: 1, pos: 213

### Candidate
type: DSZ, layer: 1, pos: 118

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392900, upper bound: 0.0392902
time: 2.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392989, upper bound: 0.0392845
time: 2.35 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0949643, 0.0846384, -0.0949643, 0.0846384, -0.1796028, 0.1796028
1: -0.0607208, 0.0474269, -0.0607208, 0.0474269, -0.1081477, 0.1081477
2: -0.1246243, 0.0634956, -0.1246243, 0.0634956, -0.1881199, 0.1881199
3: 0.9865127, 1.0434601, 0.9865127, 1.0434601, -0.0569474, 0.0569474
4: -0.0387707, 0.1049149, -0.0387707, 0.1049149, -0.1436856, 0.1436856
5: -0.0581584, 0.1370523, -0.0581584, 0.1370523, -0.1952107, 0.1952107
6: -0.1140255, 0.0998719, -0.1140255, 0.0998719, -0.2138974, 0.2138974
7: -0.0948062, 0.0074982, -0.0948062, 0.0074982, -0.1023044, 0.1023044
8: -0.0549271, 0.0979662, -0.0549271, 0.0979662, -0.1528933, 0.1528933
9: -0.0670059, 0.0810942, -0.0670059, 0.0810942, -0.1481002, 0.1481002

Time for backsubstitution: 1.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Candidate
type: DSZ, layer: 1, pos: 213

### Candidate
type: DSZ, layer: 1, pos: 118

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392810, upper bound: 0.0392866
time: 2.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393018, upper bound: 0.0392811
time: 2.33 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0949643, 0.0846384, -0.0949643, 0.0846384, -0.1796028, 0.1796028
1: -0.0607208, 0.0474269, -0.0607208, 0.0474269, -0.1081477, 0.1081477
2: -0.1246243, 0.0634956, -0.1246243, 0.0634956, -0.1881199, 0.1881199
3: 0.9865127, 1.0434601, 0.9865127, 1.0434601, -0.0569474, 0.0569474
4: -0.0387707, 0.1049149, -0.0387707, 0.1049149, -0.1436856, 0.1436856
5: -0.0581584, 0.1370523, -0.0581584, 0.1370523, -0.1952107, 0.1952107
6: -0.1140255, 0.0998719, -0.1140255, 0.0998719, -0.2138974, 0.2138974
7: -0.0948062, 0.0074982, -0.0948062, 0.0074982, -0.1023044, 0.1023044
8: -0.0549271, 0.0979662, -0.0549271, 0.0979662, -0.1528933, 0.1528933
9: -0.0670059, 0.0810942, -0.0670059, 0.0810942, -0.1481002, 0.1481002

Time for backsubstitution: 1.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Candidate
type: DSZ, layer: 1, pos: 213

### Candidate
type: DSZ, layer: 1, pos: 118

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392810, upper bound: 0.0392864
time: 2.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393020, upper bound: 0.0392810
time: 2.21 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0949643, 0.0846384, -0.0949643, 0.0846384, -0.1796028, 0.1796028
1: -0.0607208, 0.0474269, -0.0607208, 0.0474269, -0.1081477, 0.1081477
2: -0.1246243, 0.0634956, -0.1246243, 0.0634956, -0.1881199, 0.1881199
3: 0.9865127, 1.0434601, 0.9865127, 1.0434601, -0.0569474, 0.0569474
4: -0.0387707, 0.1049149, -0.0387707, 0.1049149, -0.1436856, 0.1436856
5: -0.0581584, 0.1370523, -0.0581584, 0.1370523, -0.1952107, 0.1952107
6: -0.1140255, 0.0998719, -0.1140255, 0.0998719, -0.2138974, 0.2138974
7: -0.0948062, 0.0074982, -0.0948062, 0.0074982, -0.1023044, 0.1023044
8: -0.0549271, 0.0979662, -0.0549271, 0.0979662, -0.1528933, 0.1528933
9: -0.0670059, 0.0810942, -0.0670059, 0.0810942, -0.1481002, 0.1481002

Time for backsubstitution: 1.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Candidate
type: DSZ, layer: 1, pos: 213

### Candidate
type: DSZ, layer: 1, pos: 118

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392898, upper bound: 0.0392903
time: 2.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392985, upper bound: 0.0392846
time: 2.42 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0949643, 0.0846384, -0.0949643, 0.0846384, -0.1796028, 0.1796028
1: -0.0607208, 0.0474269, -0.0607208, 0.0474269, -0.1081477, 0.1081477
2: -0.1246243, 0.0634956, -0.1246243, 0.0634956, -0.1881199, 0.1881199
3: 0.9865127, 1.0434601, 0.9865127, 1.0434601, -0.0569474, 0.0569474
4: -0.0387707, 0.1049149, -0.0387707, 0.1049149, -0.1436856, 0.1436856
5: -0.0581584, 0.1370523, -0.0581584, 0.1370523, -0.1952107, 0.1952107
6: -0.1140255, 0.0998719, -0.1140255, 0.0998719, -0.2138974, 0.2138974
7: -0.0948062, 0.0074982, -0.0948062, 0.0074982, -0.1023044, 0.1023044
8: -0.0549271, 0.0979662, -0.0549271, 0.0979662, -0.1528933, 0.1528933
9: -0.0670059, 0.0810942, -0.0670059, 0.0810942, -0.1481002, 0.1481002

Time for backsubstitution: 1.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Candidate
type: DSZ, layer: 1, pos: 213

### Candidate
type: DSZ, layer: 1, pos: 118

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392900, upper bound: 0.0392902
time: 2.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392989, upper bound: 0.0392845
time: 2.19 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 6.23 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.23
Output dim: 3, lower bound: -0.0392845, upper bound: 0.0392988
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.23
Output dim: 3, lower bound: -0.0392902, upper bound: 0.0392901
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.23
Output dim: 3, lower bound: -0.0392845, upper bound: 0.0392985
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.23
Output dim: 3, lower bound: -0.0392903, upper bound: 0.0392898
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.23
Output dim: 3, lower bound: -0.0392810, upper bound: 0.0393020
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.23
Output dim: 3, lower bound: -0.0392864, upper bound: 0.0392932
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.23
Output dim: 3, lower bound: -0.0392810, upper bound: 0.0393018
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.23
Output dim: 3, lower bound: -0.0392810, upper bound: 0.0392932
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.23
Output dim: 3, lower bound: -0.0392810, upper bound: 0.0392988
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.23
Output dim: 3, lower bound: -0.0392902, upper bound: 0.0392900
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.23
Output dim: 3, lower bound: -0.0392845, upper bound: 0.0392985
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.23
Output dim: 3, lower bound: -0.0392903, upper bound: 0.0392898
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.23
Output dim: 3, lower bound: -0.0392810, upper bound: 0.0393020
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.23
Output dim: 3, lower bound: -0.0392810, upper bound: 0.0392932
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.23
Output dim: 3, lower bound: -0.0392810, upper bound: 0.0393018
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.23
Output dim: 3, lower bound: -0.0392810, upper bound: 0.0392932
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.23
Output dim: 3, lower bound: -0.0392845, upper bound: 0.0392866
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.23
Output dim: 3, lower bound: -0.0393018, upper bound: 0.0392811
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.23
Output dim: 3, lower bound: -0.0392932, upper bound: 0.0392864
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.23
Output dim: 3, lower bound: -0.0393020, upper bound: 0.0392810
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.23
Output dim: 3, lower bound: -0.0392898, upper bound: 0.0392902
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.23
Output dim: 3, lower bound: -0.0392985, upper bound: 0.0392845
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.23
Output dim: 3, lower bound: -0.0392900, upper bound: 0.0392902
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.23
Output dim: 3, lower bound: -0.0392989, upper bound: 0.0392845
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.23
Output dim: 3, lower bound: -0.0392810, upper bound: 0.0392866
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.23
Output dim: 3, lower bound: -0.0393018, upper bound: 0.0392811
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.23
Output dim: 3, lower bound: -0.0392932, upper bound: 0.0392864
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.23
Output dim: 3, lower bound: -0.0393020, upper bound: 0.0392810
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.23
Output dim: 3, lower bound: -0.0392898, upper bound: 0.0392903
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.23
Output dim: 3, lower bound: -0.0392985, upper bound: 0.0392845
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.23
Output dim: 3, lower bound: -0.0392900, upper bound: 0.0392902
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.23
Output dim: 3, lower bound: -0.0392989, upper bound: 0.0392845
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.23
Output dim: 3, lower bound: -0.0392845, upper bound: 0.0392988
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.23
Output dim: 3, lower bound: -0.0392902, upper bound: 0.0392901
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.23
Output dim: 3, lower bound: -0.0392845, upper bound: 0.0392985
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.23
Output dim: 3, lower bound: -0.0392903, upper bound: 0.0392898
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.23
Output dim: 3, lower bound: -0.0392810, upper bound: 0.0393020
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.23
Output dim: 3, lower bound: -0.0392864, upper bound: 0.0392933
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.23
Output dim: 3, lower bound: -0.0392810, upper bound: 0.0393018
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.23
Output dim: 3, lower bound: -0.0392810, upper bound: 0.0392932
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.23
Output dim: 3, lower bound: -0.0392845, upper bound: 0.0392989
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.23
Output dim: 3, lower bound: -0.0392902, upper bound: 0.0392900
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.23
Output dim: 3, lower bound: -0.0392845, upper bound: 0.0392985
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.23
Output dim: 3, lower bound: -0.0392903, upper bound: 0.0392898
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.23
Output dim: 3, lower bound: -0.0392810, upper bound: 0.0393020
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.23
Output dim: 3, lower bound: -0.0392864, upper bound: 0.0392932
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.23
Output dim: 3, lower bound: -0.0392810, upper bound: 0.0393018
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.23
Output dim: 3, lower bound: -0.0392810, upper bound: 0.0392932
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.23
Output dim: 3, lower bound: -0.0392810, upper bound: 0.0392866
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.23
Output dim: 3, lower bound: -0.0393018, upper bound: 0.0392810
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.23
Output dim: 3, lower bound: -0.0392845, upper bound: 0.0392864
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.23
Output dim: 3, lower bound: -0.0393020, upper bound: 0.0392810
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.23
Output dim: 3, lower bound: -0.0392898, upper bound: 0.0392903
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.23
Output dim: 3, lower bound: -0.0392985, upper bound: 0.0392846
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.23
Output dim: 3, lower bound: -0.0392900, upper bound: 0.0392902
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.23
Output dim: 3, lower bound: -0.0392989, upper bound: 0.0392845
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.23
Output dim: 3, lower bound: -0.0392810, upper bound: 0.0392866
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.23
Output dim: 3, lower bound: -0.0393018, upper bound: 0.0392811
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.23
Output dim: 3, lower bound: -0.0392810, upper bound: 0.0392864
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.23
Output dim: 3, lower bound: -0.0393020, upper bound: 0.0392810
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.23
Output dim: 3, lower bound: -0.0392898, upper bound: 0.0392903
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.23
Output dim: 3, lower bound: -0.0392985, upper bound: 0.0392846
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.23
Output dim: 3, lower bound: -0.0392900, upper bound: 0.0392902
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.23
Output dim: 3, lower bound: -0.0392989, upper bound: 0.0392845

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0949643, 0.0846384, -0.0949643, 0.0846384, -0.1796028, 0.1796028
1: -0.0607208, 0.0474269, -0.0607208, 0.0474269, -0.1081477, 0.1081477
2: -0.1246243, 0.0634956, -0.1246243, 0.0634956, -0.1881199, 0.1881199
3: 0.9865127, 1.0434601, 0.9865127, 1.0434601, -0.0569474, 0.0569474
4: -0.0387707, 0.1049149, -0.0387707, 0.1049149, -0.1436856, 0.1436856
5: -0.0581584, 0.1370523, -0.0581584, 0.1370523, -0.1952107, 0.1952107
6: -0.1140255, 0.0998719, -0.1140255, 0.0998719, -0.2138974, 0.2138974
7: -0.0948062, 0.0074982, -0.0948062, 0.0074982, -0.1023044, 0.1023044
8: -0.0549271, 0.0979662, -0.0549271, 0.0979662, -0.1528933, 0.1528933
9: -0.0670059, 0.0810942, -0.0670059, 0.0810942, -0.1481002, 0.1481002

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Candidate
type: DSZ, layer: 1, pos: 213

### Candidate
type: DSZ, layer: 1, pos: 212

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392707, upper bound: 0.0392880
time: 3.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392743, upper bound: 0.0392880
time: 2.55 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0949643, 0.0846384, -0.0949643, 0.0846384, -0.1796028, 0.1796028
1: -0.0607208, 0.0474269, -0.0607208, 0.0474269, -0.1081477, 0.1081477
2: -0.1246243, 0.0634956, -0.1246243, 0.0634956, -0.1881199, 0.1881199
3: 0.9865127, 1.0434601, 0.9865127, 1.0434601, -0.0569474, 0.0569474
4: -0.0387707, 0.1049149, -0.0387707, 0.1049149, -0.1436856, 0.1436856
5: -0.0581584, 0.1370523, -0.0581584, 0.1370523, -0.1952107, 0.1952107
6: -0.1140255, 0.0998719, -0.1140255, 0.0998719, -0.2138974, 0.2138974
7: -0.0948062, 0.0074982, -0.0948062, 0.0074982, -0.1023044, 0.1023044
8: -0.0549271, 0.0979662, -0.0549271, 0.0979662, -0.1528933, 0.1528933
9: -0.0670059, 0.0810942, -0.0670059, 0.0810942, -0.1481002, 0.1481002

Time for backsubstitution: 1.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Candidate
type: DSZ, layer: 1, pos: 213

### Candidate
type: DSZ, layer: 1, pos: 212

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392800, upper bound: 0.0392794
time: 2.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392800, upper bound: 0.0392794
time: 2.79 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0949643, 0.0846384, -0.0949643, 0.0846384, -0.1796028, 0.1796028
1: -0.0607208, 0.0474269, -0.0607208, 0.0474269, -0.1081477, 0.1081477
2: -0.1246243, 0.0634956, -0.1246243, 0.0634956, -0.1881199, 0.1881199
3: 0.9865127, 1.0434601, 0.9865127, 1.0434601, -0.0569474, 0.0569474
4: -0.0387707, 0.1049149, -0.0387707, 0.1049149, -0.1436856, 0.1436856
5: -0.0581584, 0.1370523, -0.0581584, 0.1370523, -0.1952107, 0.1952107
6: -0.1140255, 0.0998719, -0.1140255, 0.0998719, -0.2138974, 0.2138974
7: -0.0948062, 0.0074982, -0.0948062, 0.0074982, -0.1023044, 0.1023044
8: -0.0549271, 0.0979662, -0.0549271, 0.0979662, -0.1528933, 0.1528933
9: -0.0670059, 0.0810942, -0.0670059, 0.0810942, -0.1481002, 0.1481002

Time for backsubstitution: 1.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Candidate
type: DSZ, layer: 1, pos: 213

### Candidate
type: DSZ, layer: 1, pos: 212

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392744, upper bound: 0.0392875
time: 5.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392707, upper bound: 0.0392875
time: 2.42 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0949643, 0.0846384, -0.0949643, 0.0846384, -0.1796028, 0.1796028
1: -0.0607208, 0.0474269, -0.0607208, 0.0474269, -0.1081477, 0.1081477
2: -0.1246243, 0.0634956, -0.1246243, 0.0634956, -0.1881199, 0.1881199
3: 0.9865127, 1.0434601, 0.9865127, 1.0434601, -0.0569474, 0.0569474
4: -0.0387707, 0.1049149, -0.0387707, 0.1049149, -0.1436856, 0.1436856
5: -0.0581584, 0.1370523, -0.0581584, 0.1370523, -0.1952107, 0.1952107
6: -0.1140255, 0.0998719, -0.1140255, 0.0998719, -0.2138974, 0.2138974
7: -0.0948062, 0.0074982, -0.0948062, 0.0074982, -0.1023044, 0.1023044
8: -0.0549271, 0.0979662, -0.0549271, 0.0979662, -0.1528933, 0.1528933
9: -0.0670059, 0.0810942, -0.0670059, 0.0810942, -0.1481002, 0.1481002

Time for backsubstitution: 1.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Candidate
type: DSZ, layer: 1, pos: 213

### Candidate
type: DSZ, layer: 1, pos: 212

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392801, upper bound: 0.0392791
time: 2.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392707, upper bound: 0.0392791
time: 2.66 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0949643, 0.0846384, -0.0949643, 0.0846384, -0.1796028, 0.1796028
1: -0.0607208, 0.0474269, -0.0607208, 0.0474269, -0.1081477, 0.1081477
2: -0.1246243, 0.0634956, -0.1246243, 0.0634956, -0.1881199, 0.1881199
3: 0.9865127, 1.0434601, 0.9865127, 1.0434601, -0.0569474, 0.0569474
4: -0.0387707, 0.1049149, -0.0387707, 0.1049149, -0.1436856, 0.1436856
5: -0.0581584, 0.1370523, -0.0581584, 0.1370523, -0.1952107, 0.1952107
6: -0.1140255, 0.0998719, -0.1140255, 0.0998719, -0.2138974, 0.2138974
7: -0.0948062, 0.0074982, -0.0948062, 0.0074982, -0.1023044, 0.1023044
8: -0.0549271, 0.0979662, -0.0549271, 0.0979662, -0.1528933, 0.1528933
9: -0.0670059, 0.0810942, -0.0670059, 0.0810942, -0.1481002, 0.1481002

Time for backsubstitution: 1.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Candidate
type: DSZ, layer: 1, pos: 213

### Candidate
type: DSZ, layer: 1, pos: 212

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392707, upper bound: 0.0392910
time: 2.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392707, upper bound: 0.0392910
time: 2.01 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0949643, 0.0846384, -0.0949643, 0.0846384, -0.1796028, 0.1796028
1: -0.0607208, 0.0474269, -0.0607208, 0.0474269, -0.1081477, 0.1081477
2: -0.1246243, 0.0634956, -0.1246243, 0.0634956, -0.1881199, 0.1881199
3: 0.9865127, 1.0434601, 0.9865127, 1.0434601, -0.0569474, 0.0569474
4: -0.0387707, 0.1049149, -0.0387707, 0.1049149, -0.1436856, 0.1436856
5: -0.0581584, 0.1370523, -0.0581584, 0.1370523, -0.1952107, 0.1952107
6: -0.1140255, 0.0998719, -0.1140255, 0.0998719, -0.2138974, 0.2138974
7: -0.0948062, 0.0074982, -0.0948062, 0.0074982, -0.1023044, 0.1023044
8: -0.0549271, 0.0979662, -0.0549271, 0.0979662, -0.1528933, 0.1528933
9: -0.0670059, 0.0810942, -0.0670059, 0.0810942, -0.1481002, 0.1481002

Time for backsubstitution: 1.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Candidate
type: DSZ, layer: 1, pos: 213

### Candidate
type: DSZ, layer: 1, pos: 212

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392760, upper bound: 0.0392824
time: 2.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392707, upper bound: 0.0392824
time: 2.85 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0949643, 0.0846384, -0.0949643, 0.0846384, -0.1796028, 0.1796028
1: -0.0607208, 0.0474269, -0.0607208, 0.0474269, -0.1081477, 0.1081477
2: -0.1246243, 0.0634956, -0.1246243, 0.0634956, -0.1881199, 0.1881199
3: 0.9865127, 1.0434601, 0.9865127, 1.0434601, -0.0569474, 0.0569474
4: -0.0387707, 0.1049149, -0.0387707, 0.1049149, -0.1436856, 0.1436856
5: -0.0581584, 0.1370523, -0.0581584, 0.1370523, -0.1952107, 0.1952107
6: -0.1140255, 0.0998719, -0.1140255, 0.0998719, -0.2138974, 0.2138974
7: -0.0948062, 0.0074982, -0.0948062, 0.0074982, -0.1023044, 0.1023044
8: -0.0549271, 0.0979662, -0.0549271, 0.0979662, -0.1528933, 0.1528933
9: -0.0670059, 0.0810942, -0.0670059, 0.0810942, -0.1481002, 0.1481002

Time for backsubstitution: 1.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Candidate
type: DSZ, layer: 1, pos: 213

### Candidate
type: DSZ, layer: 1, pos: 212

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392707, upper bound: 0.0392909
time: 2.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392707, upper bound: 0.0392909
time: 2.48 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0949643, 0.0846384, -0.0949643, 0.0846384, -0.1796028, 0.1796028
1: -0.0607208, 0.0474269, -0.0607208, 0.0474269, -0.1081477, 0.1081477
2: -0.1246243, 0.0634956, -0.1246243, 0.0634956, -0.1881199, 0.1881199
3: 0.9865127, 1.0434601, 0.9865127, 1.0434601, -0.0569474, 0.0569474
4: -0.0387707, 0.1049149, -0.0387707, 0.1049149, -0.1436856, 0.1436856
5: -0.0581584, 0.1370523, -0.0581584, 0.1370523, -0.1952107, 0.1952107
6: -0.1140255, 0.0998719, -0.1140255, 0.0998719, -0.2138974, 0.2138974
7: -0.0948062, 0.0074982, -0.0948062, 0.0074982, -0.1023044, 0.1023044
8: -0.0549271, 0.0979662, -0.0549271, 0.0979662, -0.1528933, 0.1528933
9: -0.0670059, 0.0810942, -0.0670059, 0.0810942, -0.1481002, 0.1481002

Time for backsubstitution: 1.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Candidate
type: DSZ, layer: 1, pos: 213

### Candidate
type: DSZ, layer: 1, pos: 212

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392762, upper bound: 0.0392823
time: 2.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392707, upper bound: 0.0392823
time: 3.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0949643, 0.0846384, -0.0949643, 0.0846384, -0.1796028, 0.1796028
1: -0.0607208, 0.0474269, -0.0607208, 0.0474269, -0.1081477, 0.1081477
2: -0.1246243, 0.0634956, -0.1246243, 0.0634956, -0.1881199, 0.1881199
3: 0.9865127, 1.0434601, 0.9865127, 1.0434601, -0.0569474, 0.0569474
4: -0.0387707, 0.1049149, -0.0387707, 0.1049149, -0.1436856, 0.1436856
5: -0.0581584, 0.1370523, -0.0581584, 0.1370523, -0.1952107, 0.1952107
6: -0.1140255, 0.0998719, -0.1140255, 0.0998719, -0.2138974, 0.2138974
7: -0.0948062, 0.0074982, -0.0948062, 0.0074982, -0.1023044, 0.1023044
8: -0.0549271, 0.0979662, -0.0549271, 0.0979662, -0.1528933, 0.1528933
9: -0.0670059, 0.0810942, -0.0670059, 0.0810942, -0.1481002, 0.1481002

Time for backsubstitution: 1.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Candidate
type: DSZ, layer: 1, pos: 213

### Candidate
type: DSZ, layer: 1, pos: 212

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392707, upper bound: 0.0392880
time: 2.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392707, upper bound: 0.0392880
time: 2.26 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0949643, 0.0846384, -0.0949643, 0.0846384, -0.1796028, 0.1796028
1: -0.0607208, 0.0474269, -0.0607208, 0.0474269, -0.1081477, 0.1081477
2: -0.1246243, 0.0634956, -0.1246243, 0.0634956, -0.1881199, 0.1881199
3: 0.9865127, 1.0434601, 0.9865127, 1.0434601, -0.0569474, 0.0569474
4: -0.0387707, 0.1049149, -0.0387707, 0.1049149, -0.1436856, 0.1436856
5: -0.0581584, 0.1370523, -0.0581584, 0.1370523, -0.1952107, 0.1952107
6: -0.1140255, 0.0998719, -0.1140255, 0.0998719, -0.2138974, 0.2138974
7: -0.0948062, 0.0074982, -0.0948062, 0.0074982, -0.1023044, 0.1023044
8: -0.0549271, 0.0979662, -0.0549271, 0.0979662, -0.1528933, 0.1528933
9: -0.0670059, 0.0810942, -0.0670059, 0.0810942, -0.1481002, 0.1481002

Time for backsubstitution: 1.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Candidate
type: DSZ, layer: 1, pos: 213

### Candidate
type: DSZ, layer: 1, pos: 212

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392707, upper bound: 0.0392794
time: 5.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392800, upper bound: 0.0392794
time: 3.00 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0949643, 0.0846384, -0.0949643, 0.0846384, -0.1796028, 0.1796028
1: -0.0607208, 0.0474269, -0.0607208, 0.0474269, -0.1081477, 0.1081477
2: -0.1246243, 0.0634956, -0.1246243, 0.0634956, -0.1881199, 0.1881199
3: 0.9865127, 1.0434601, 0.9865127, 1.0434601, -0.0569474, 0.0569474
4: -0.0387707, 0.1049149, -0.0387707, 0.1049149, -0.1436856, 0.1436856
5: -0.0581584, 0.1370523, -0.0581584, 0.1370523, -0.1952107, 0.1952107
6: -0.1140255, 0.0998719, -0.1140255, 0.0998719, -0.2138974, 0.2138974
7: -0.0948062, 0.0074982, -0.0948062, 0.0074982, -0.1023044, 0.1023044
8: -0.0549271, 0.0979662, -0.0549271, 0.0979662, -0.1528933, 0.1528933
9: -0.0670059, 0.0810942, -0.0670059, 0.0810942, -0.1481002, 0.1481002

Time for backsubstitution: 1.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Candidate
type: DSZ, layer: 1, pos: 213

### Candidate
type: DSZ, layer: 1, pos: 212

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392707, upper bound: 0.0392875
time: 2.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392707, upper bound: 0.0392875
time: 2.34 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0949643, 0.0846384, -0.0949643, 0.0846384, -0.1796028, 0.1796028
1: -0.0607208, 0.0474269, -0.0607208, 0.0474269, -0.1081477, 0.1081477
2: -0.1246243, 0.0634956, -0.1246243, 0.0634956, -0.1881199, 0.1881199
3: 0.9865127, 1.0434601, 0.9865127, 1.0434601, -0.0569474, 0.0569474
4: -0.0387707, 0.1049149, -0.0387707, 0.1049149, -0.1436856, 0.1436856
5: -0.0581584, 0.1370523, -0.0581584, 0.1370523, -0.1952107, 0.1952107
6: -0.1140255, 0.0998719, -0.1140255, 0.0998719, -0.2138974, 0.2138974
7: -0.0948062, 0.0074982, -0.0948062, 0.0074982, -0.1023044, 0.1023044
8: -0.0549271, 0.0979662, -0.0549271, 0.0979662, -0.1528933, 0.1528933
9: -0.0670059, 0.0810942, -0.0670059, 0.0810942, -0.1481002, 0.1481002

Time for backsubstitution: 1.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Candidate
type: DSZ, layer: 1, pos: 213

### Candidate
type: DSZ, layer: 1, pos: 212

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392801, upper bound: 0.0392791
time: 2.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392801, upper bound: 0.0392791
time: 2.63 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0949643, 0.0846384, -0.0949643, 0.0846384, -0.1796028, 0.1796028
1: -0.0607208, 0.0474269, -0.0607208, 0.0474269, -0.1081477, 0.1081477
2: -0.1246243, 0.0634956, -0.1246243, 0.0634956, -0.1881199, 0.1881199
3: 0.9865127, 1.0434601, 0.9865127, 1.0434601, -0.0569474, 0.0569474
4: -0.0387707, 0.1049149, -0.0387707, 0.1049149, -0.1436856, 0.1436856
5: -0.0581584, 0.1370523, -0.0581584, 0.1370523, -0.1952107, 0.1952107
6: -0.1140255, 0.0998719, -0.1140255, 0.0998719, -0.2138974, 0.2138974
7: -0.0948062, 0.0074982, -0.0948062, 0.0074982, -0.1023044, 0.1023044
8: -0.0549271, 0.0979662, -0.0549271, 0.0979662, -0.1528933, 0.1528933
9: -0.0670059, 0.0810942, -0.0670059, 0.0810942, -0.1481002, 0.1481002

Time for backsubstitution: 1.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Candidate
type: DSZ, layer: 1, pos: 213

### Candidate
type: DSZ, layer: 1, pos: 212

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392707, upper bound: 0.0392910
time: 2.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392707, upper bound: 0.0392910
time: 2.38 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0949643, 0.0846384, -0.0949643, 0.0846384, -0.1796028, 0.1796028
1: -0.0607208, 0.0474269, -0.0607208, 0.0474269, -0.1081477, 0.1081477
2: -0.1246243, 0.0634956, -0.1246243, 0.0634956, -0.1881199, 0.1881199
3: 0.9865127, 1.0434601, 0.9865127, 1.0434601, -0.0569474, 0.0569474
4: -0.0387707, 0.1049149, -0.0387707, 0.1049149, -0.1436856, 0.1436856
5: -0.0581584, 0.1370523, -0.0581584, 0.1370523, -0.1952107, 0.1952107
6: -0.1140255, 0.0998719, -0.1140255, 0.0998719, -0.2138974, 0.2138974
7: -0.0948062, 0.0074982, -0.0948062, 0.0074982, -0.1023044, 0.1023044
8: -0.0549271, 0.0979662, -0.0549271, 0.0979662, -0.1528933, 0.1528933
9: -0.0670059, 0.0810942, -0.0670059, 0.0810942, -0.1481002, 0.1481002

Time for backsubstitution: 1.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Candidate
type: DSZ, layer: 1, pos: 213

### Candidate
type: DSZ, layer: 1, pos: 212

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392707, upper bound: 0.0392824
time: 2.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392707, upper bound: 0.0392824
time: 2.28 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0949643, 0.0846384, -0.0949643, 0.0846384, -0.1796028, 0.1796028
1: -0.0607208, 0.0474269, -0.0607208, 0.0474269, -0.1081477, 0.1081477
2: -0.1246243, 0.0634956, -0.1246243, 0.0634956, -0.1881199, 0.1881199
3: 0.9865127, 1.0434601, 0.9865127, 1.0434601, -0.0569474, 0.0569474
4: -0.0387707, 0.1049149, -0.0387707, 0.1049149, -0.1436856, 0.1436856
5: -0.0581584, 0.1370523, -0.0581584, 0.1370523, -0.1952107, 0.1952107
6: -0.1140255, 0.0998719, -0.1140255, 0.0998719, -0.2138974, 0.2138974
7: -0.0948062, 0.0074982, -0.0948062, 0.0074982, -0.1023044, 0.1023044
8: -0.0549271, 0.0979662, -0.0549271, 0.0979662, -0.1528933, 0.1528933
9: -0.0670059, 0.0810942, -0.0670059, 0.0810942, -0.1481002, 0.1481002

Time for backsubstitution: 1.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Candidate
type: DSZ, layer: 1, pos: 213

### Candidate
type: DSZ, layer: 1, pos: 212

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392707, upper bound: 0.0392908
time: 3.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392707, upper bound: 0.0392908
time: 2.32 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 7.43 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 7.43
Output dim: 3, lower bound: -0.0392707, upper bound: 0.0392880
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 7.43
Output dim: 3, lower bound: -0.0392743, upper bound: 0.0392880
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 7.43
Output dim: 3, lower bound: -0.0392800, upper bound: 0.0392794
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 7.43
Output dim: 3, lower bound: -0.0392800, upper bound: 0.0392794
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 7.43
Output dim: 3, lower bound: -0.0392744, upper bound: 0.0392875
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 7.43
Output dim: 3, lower bound: -0.0392707, upper bound: 0.0392875
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 7.43
Output dim: 3, lower bound: -0.0392801, upper bound: 0.0392791
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 7.43
Output dim: 3, lower bound: -0.0392707, upper bound: 0.0392791
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 7.43
Output dim: 3, lower bound: -0.0392707, upper bound: 0.0392910
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 7.43
Output dim: 3, lower bound: -0.0392707, upper bound: 0.0392910
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 7.43
Output dim: 3, lower bound: -0.0392760, upper bound: 0.0392824
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 7.43
Output dim: 3, lower bound: -0.0392707, upper bound: 0.0392824
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 7.43
Output dim: 3, lower bound: -0.0392707, upper bound: 0.0392909
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 7.43
Output dim: 3, lower bound: -0.0392707, upper bound: 0.0392909
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 7.43
Output dim: 3, lower bound: -0.0392762, upper bound: 0.0392823
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 7.43
Output dim: 3, lower bound: -0.0392707, upper bound: 0.0392823
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 7.43
Output dim: 3, lower bound: -0.0392707, upper bound: 0.0392880
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 7.43
Output dim: 3, lower bound: -0.0392707, upper bound: 0.0392880
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 7.43
Output dim: 3, lower bound: -0.0392707, upper bound: 0.0392794
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 7.43
Output dim: 3, lower bound: -0.0392800, upper bound: 0.0392794
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 7.43
Output dim: 3, lower bound: -0.0392707, upper bound: 0.0392875
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 7.43
Output dim: 3, lower bound: -0.0392707, upper bound: 0.0392875
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 7.43
Output dim: 3, lower bound: -0.0392801, upper bound: 0.0392791
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 7.43
Output dim: 3, lower bound: -0.0392801, upper bound: 0.0392791
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 7.43
Output dim: 3, lower bound: -0.0392707, upper bound: 0.0392910
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 7.43
Output dim: 3, lower bound: -0.0392707, upper bound: 0.0392910
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 7.43
Output dim: 3, lower bound: -0.0392707, upper bound: 0.0392824
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 7.43
Output dim: 3, lower bound: -0.0392707, upper bound: 0.0392824
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 7.43
Output dim: 3, lower bound: -0.0392707, upper bound: 0.0392908
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 7.43
Output dim: 3, lower bound: -0.0392707, upper bound: 0.0392908
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 7.43
Output dim: 3, lower bound: -0.0392810, upper bound: 0.0392932
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 7.43
Output dim: 3, lower bound: -0.0392845, upper bound: 0.0392866
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 7.43
Output dim: 3, lower bound: -0.0393018, upper bound: 0.0392811
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 7.43
Output dim: 3, lower bound: -0.0392932, upper bound: 0.0392864
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 7.43
Output dim: 3, lower bound: -0.0393020, upper bound: 0.0392810
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 7.43
Output dim: 3, lower bound: -0.0392898, upper bound: 0.0392902
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 7.43
Output dim: 3, lower bound: -0.0392985, upper bound: 0.0392845
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 7.43
Output dim: 3, lower bound: -0.0392900, upper bound: 0.0392902
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 7.43
Output dim: 3, lower bound: -0.0392989, upper bound: 0.0392845
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 7.43
Output dim: 3, lower bound: -0.0392810, upper bound: 0.0392866
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 7.43
Output dim: 3, lower bound: -0.0393018, upper bound: 0.0392811
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 7.43
Output dim: 3, lower bound: -0.0392932, upper bound: 0.0392864
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 7.43
Output dim: 3, lower bound: -0.0393020, upper bound: 0.0392810
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 7.43
Output dim: 3, lower bound: -0.0392898, upper bound: 0.0392903
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 7.43
Output dim: 3, lower bound: -0.0392985, upper bound: 0.0392845
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 7.43
Output dim: 3, lower bound: -0.0392900, upper bound: 0.0392902
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 7.43
Output dim: 3, lower bound: -0.0392989, upper bound: 0.0392845
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 7.43
Output dim: 3, lower bound: -0.0392845, upper bound: 0.0392988
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 7.43
Output dim: 3, lower bound: -0.0392902, upper bound: 0.0392901
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 7.43
Output dim: 3, lower bound: -0.0392845, upper bound: 0.0392985
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 7.43
Output dim: 3, lower bound: -0.0392903, upper bound: 0.0392898
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 7.43
Output dim: 3, lower bound: -0.0392810, upper bound: 0.0393020
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 7.43
Output dim: 3, lower bound: -0.0392864, upper bound: 0.0392933
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 7.43
Output dim: 3, lower bound: -0.0392810, upper bound: 0.0393018
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 7.43
Output dim: 3, lower bound: -0.0392810, upper bound: 0.0392932
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 7.43
Output dim: 3, lower bound: -0.0392845, upper bound: 0.0392989
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 7.43
Output dim: 3, lower bound: -0.0392902, upper bound: 0.0392900
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 7.43
Output dim: 3, lower bound: -0.0392845, upper bound: 0.0392985
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 7.43
Output dim: 3, lower bound: -0.0392903, upper bound: 0.0392898
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 7.43
Output dim: 3, lower bound: -0.0392810, upper bound: 0.0393020
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 7.43
Output dim: 3, lower bound: -0.0392864, upper bound: 0.0392932
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 7.43
Output dim: 3, lower bound: -0.0392810, upper bound: 0.0393018
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 7.43
Output dim: 3, lower bound: -0.0392810, upper bound: 0.0392932
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 7.43
Output dim: 3, lower bound: -0.0392810, upper bound: 0.0392866
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 7.43
Output dim: 3, lower bound: -0.0393018, upper bound: 0.0392810
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 7.43
Output dim: 3, lower bound: -0.0392845, upper bound: 0.0392864
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 7.43
Output dim: 3, lower bound: -0.0393020, upper bound: 0.0392810
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 7.43
Output dim: 3, lower bound: -0.0392898, upper bound: 0.0392903
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 7.43
Output dim: 3, lower bound: -0.0392985, upper bound: 0.0392846
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 7.43
Output dim: 3, lower bound: -0.0392900, upper bound: 0.0392902
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 7.43
Output dim: 3, lower bound: -0.0392989, upper bound: 0.0392845
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 7.43
Output dim: 3, lower bound: -0.0392810, upper bound: 0.0392866
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 7.43
Output dim: 3, lower bound: -0.0393018, upper bound: 0.0392811
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 7.43
Output dim: 3, lower bound: -0.0392810, upper bound: 0.0392864
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 7.43
Output dim: 3, lower bound: -0.0393020, upper bound: 0.0392810
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 7.43
Output dim: 3, lower bound: -0.0392898, upper bound: 0.0392903
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 7.43
Output dim: 3, lower bound: -0.0392985, upper bound: 0.0392846
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 7.43
Output dim: 3, lower bound: -0.0392900, upper bound: 0.0392902
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 7.43
Output dim: 3, lower bound: -0.0392989, upper bound: 0.0392845

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 4.67 + 600.77 = 605.43 seconds
