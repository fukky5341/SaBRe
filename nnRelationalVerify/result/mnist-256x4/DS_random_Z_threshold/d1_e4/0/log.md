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
execution time: IAR + RelationalAnalysis = 0.93 + 3.05 = 3.98 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.0408667, upper bound: 0.0408667

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 255

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0405521, upper bound: 0.0405521
time: 3.43 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0405521, upper bound: 0.0405521
time: 2.42 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 5.86 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 5.86
Output dim: 3, lower bound: -0.0405521, upper bound: 0.0405521
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 5.86
Output dim: 3, lower bound: -0.0405521, upper bound: 0.0405521

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

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 107

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0402556, upper bound: 0.0402792
time: 2.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0402791, upper bound: 0.0402556
time: 2.46 seconds

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

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0403757, upper bound: 0.0403756
time: 2.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0403757, upper bound: 0.0403756
time: 2.30 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 5.60 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 5.60
Output dim: 3, lower bound: -0.0402556, upper bound: 0.0402792
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 5.60
Output dim: 3, lower bound: -0.0402791, upper bound: 0.0402556
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 5.60
Output dim: 3, lower bound: -0.0403757, upper bound: 0.0403756
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 5.60
Output dim: 3, lower bound: -0.0403757, upper bound: 0.0403756

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

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0400689, upper bound: 0.0400923
time: 2.36 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0400689, upper bound: 0.0400923
time: 2.10 seconds

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

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 88

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0400222, upper bound: 0.0400141
time: 5.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0400401, upper bound: 0.0400083
time: 2.34 seconds

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

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0383934, upper bound: 0.0383938
time: 1.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0383934, upper bound: 0.0383938
time: 2.02 seconds

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

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 200

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0389668, upper bound: 0.0389667
time: 1.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0389668, upper bound: 0.0389667
time: 1.87 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 4.59 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 4.59
Output dim: 3, lower bound: -0.0400689, upper bound: 0.0400923
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 4.59
Output dim: 3, lower bound: -0.0400689, upper bound: 0.0400923
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 4.59
Output dim: 3, lower bound: -0.0400222, upper bound: 0.0400141
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 4.59
Output dim: 3, lower bound: -0.0400401, upper bound: 0.0400083
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 4.59
Output dim: 3, lower bound: -0.0383934, upper bound: 0.0383938
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 4.59
Output dim: 3, lower bound: -0.0383934, upper bound: 0.0383938
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 4.59
Output dim: 3, lower bound: -0.0389668, upper bound: 0.0389667
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 4.59
Output dim: 3, lower bound: -0.0389668, upper bound: 0.0389667

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

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 229

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0386813, upper bound: 0.0386991
time: 2.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0386813, upper bound: 0.0386991
time: 2.44 seconds

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

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 156

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0398502, upper bound: 0.0398741
time: 2.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0398502, upper bound: 0.0398741
time: 2.28 seconds

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

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 108

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0399945, upper bound: 0.0399865
time: 2.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0399945, upper bound: 0.0399865
time: 2.21 seconds

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

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 156

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0388756, upper bound: 0.0388475
time: 2.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0388756, upper bound: 0.0388475
time: 2.49 seconds

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

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 78

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0382709, upper bound: 0.0382713
time: 2.09 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0382709, upper bound: 0.0382713
time: 1.51 seconds

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

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 212

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0383923, upper bound: 0.0383927
time: 1.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0383922, upper bound: 0.0383927
time: 1.65 seconds

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

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 78

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0388558, upper bound: 0.0388559
time: 2.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0388558, upper bound: 0.0388557
time: 2.16 seconds

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

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 118

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 78

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0388558, upper bound: 0.0388559
time: 2.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0388558, upper bound: 0.0388557
time: 2.16 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 5.33 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.33
Output dim: 3, lower bound: -0.0386813, upper bound: 0.0386991
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.33
Output dim: 3, lower bound: -0.0386813, upper bound: 0.0386991
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.33
Output dim: 3, lower bound: -0.0398502, upper bound: 0.0398741
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.33
Output dim: 3, lower bound: -0.0398502, upper bound: 0.0398741
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.33
Output dim: 3, lower bound: -0.0399945, upper bound: 0.0399865
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.33
Output dim: 3, lower bound: -0.0399945, upper bound: 0.0399865
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.33
Output dim: 3, lower bound: -0.0388756, upper bound: 0.0388475
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.33
Output dim: 3, lower bound: -0.0388756, upper bound: 0.0388475
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.33
Output dim: 3, lower bound: -0.0382709, upper bound: 0.0382713
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.33
Output dim: 3, lower bound: -0.0382709, upper bound: 0.0382713
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.33
Output dim: 3, lower bound: -0.0383923, upper bound: 0.0383927
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.33
Output dim: 3, lower bound: -0.0383922, upper bound: 0.0383927
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.33
Output dim: 3, lower bound: -0.0388558, upper bound: 0.0388559
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.33
Output dim: 3, lower bound: -0.0388558, upper bound: 0.0388557
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.33
Output dim: 3, lower bound: -0.0388558, upper bound: 0.0388559
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.33
Output dim: 3, lower bound: -0.0388558, upper bound: 0.0388557

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

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 175

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0386260, upper bound: 0.0386436
time: 1.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0386258, upper bound: 0.0386437
time: 1.97 seconds

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

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 200

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 229

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0386780, upper bound: 0.0386917
time: 1.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0386733, upper bound: 0.0386961
time: 2.73 seconds

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

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0398054, upper bound: 0.0398357
time: 2.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0398054, upper bound: 0.0398213
time: 2.57 seconds

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

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 72

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0385113, upper bound: 0.0385291
time: 15.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0385113, upper bound: 0.0385291
time: 19.34 seconds

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

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 175

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0399006, upper bound: 0.0399061
time: 2.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0399144, upper bound: 0.0399063
time: 2.26 seconds

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

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 61

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0399167, upper bound: 0.0399231
time: 2.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0399167, upper bound: 0.0399231
time: 2.49 seconds

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

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 72

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 200

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0363306, upper bound: 0.0363290
time: 1.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0363306, upper bound: 0.0363290
time: 1.16 seconds

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

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 118

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0388719, upper bound: 0.0388399
time: 2.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0388669, upper bound: 0.0388438
time: 1.94 seconds

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

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 156

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 88

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0379129, upper bound: 0.0379165
time: 1.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0379162, upper bound: 0.0379127
time: 3.04 seconds

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

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 118

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 88

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0379120, upper bound: 0.0379168
time: 1.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0379159, upper bound: 0.0379137
time: 1.57 seconds

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

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 108

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 88

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0380305, upper bound: 0.0380339
time: 1.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0380332, upper bound: 0.0380312
time: 1.77 seconds

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

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Candidate
type: DSZ, layer: 1, pos: 188

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 61

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0383271, upper bound: 0.0383276
time: 2.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0383271, upper bound: 0.0383275
time: 2.50 seconds

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

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 207

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0386175, upper bound: 0.0386172
time: 1.90 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0386175, upper bound: 0.0386172
time: 2.10 seconds

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

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 112

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0388547, upper bound: 0.0388501
time: 2.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0388504, upper bound: 0.0388547
time: 1.61 seconds

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

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 212

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0388546, upper bound: 0.0388547
time: 1.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0388546, upper bound: 0.0388547
time: 1.93 seconds

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

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 88

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0385492, upper bound: 0.0385566
time: 2.09 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0385565, upper bound: 0.0385496
time: 1.96 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 4.91 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -0.0386260, upper bound: 0.0386436
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -0.0386258, upper bound: 0.0386437
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -0.0386780, upper bound: 0.0386917
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -0.0386733, upper bound: 0.0386961
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -0.0398054, upper bound: 0.0398357
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -0.0398054, upper bound: 0.0398213
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -0.0385113, upper bound: 0.0385291
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -0.0385113, upper bound: 0.0385291
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -0.0399006, upper bound: 0.0399061
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -0.0399144, upper bound: 0.0399063
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -0.0399167, upper bound: 0.0399231
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -0.0399167, upper bound: 0.0399231
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 4.91
Output dim: 3, lower bound: -0.0363306, upper bound: 0.0363290
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 4.91
Output dim: 3, lower bound: -0.0363306, upper bound: 0.0363290
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -0.0388719, upper bound: 0.0388399
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -0.0388669, upper bound: 0.0388438
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -0.0379129, upper bound: 0.0379165
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -0.0379162, upper bound: 0.0379127
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -0.0379120, upper bound: 0.0379168
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -0.0379159, upper bound: 0.0379137
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -0.0380305, upper bound: 0.0380339
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -0.0380332, upper bound: 0.0380312
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -0.0383271, upper bound: 0.0383276
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -0.0383271, upper bound: 0.0383275
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -0.0386175, upper bound: 0.0386172
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -0.0386175, upper bound: 0.0386172
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -0.0388547, upper bound: 0.0388501
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -0.0388504, upper bound: 0.0388547
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -0.0388546, upper bound: 0.0388547
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -0.0388546, upper bound: 0.0388547
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -0.0385492, upper bound: 0.0385566
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -0.0385565, upper bound: 0.0385496

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

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 229

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 118

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0385543, upper bound: 0.0385731
time: 2.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0385551, upper bound: 0.0385651
time: 1.92 seconds

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

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 61

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0385527, upper bound: 0.0385713
time: 2.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0385523, upper bound: 0.0385712
time: 1.84 seconds

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

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 78

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0385621, upper bound: 0.0385761
time: 2.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0385621, upper bound: 0.0385762
time: 1.98 seconds

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

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0385438, upper bound: 0.0385653
time: 1.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0385439, upper bound: 0.0385653
time: 1.80 seconds

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

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 188

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0391097, upper bound: 0.0391381
time: 2.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0391098, upper bound: 0.0391381
time: 2.14 seconds

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

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 134

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0398025, upper bound: 0.0398182
time: 2.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0398088, upper bound: 0.0398177
time: 2.21 seconds

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

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 118

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0365935, upper bound: 0.0365990
time: 1.13 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0365935, upper bound: 0.0365990
time: 1.15 seconds

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

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 72

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0384712, upper bound: 0.0384932
time: 1.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0384754, upper bound: 0.0384841
time: 1.88 seconds

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

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0390021, upper bound: 0.0389944
time: 2.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0390021, upper bound: 0.0389944
time: 2.23 seconds

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

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0398638, upper bound: 0.0398700
time: 2.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0398777, upper bound: 0.0398733
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

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 207

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 229

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0399267, upper bound: 0.0399147
time: 3.01 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0399074, upper bound: 0.0399190
time: 2.35 seconds

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

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0381030, upper bound: 0.0380989
time: 1.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0381030, upper bound: 0.0380989
time: 1.82 seconds

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

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 78

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0387532, upper bound: 0.0387215
time: 2.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0387532, upper bound: 0.0387215
time: 2.24 seconds

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

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 156

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 229

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0388636, upper bound: 0.0388353
time: 2.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0388583, upper bound: 0.0388404
time: 2.07 seconds

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

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 229

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 199

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0377250, upper bound: 0.0377302
time: 1.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0377271, upper bound: 0.0377282
time: 1.78 seconds

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

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 156

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 212

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0379153, upper bound: 0.0379117
time: 2.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0379152, upper bound: 0.0379117
time: 1.77 seconds

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

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 72

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 108

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0378832, upper bound: 0.0378881
time: 1.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0378832, upper bound: 0.0378881
time: 2.83 seconds

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

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0378070, upper bound: 0.0378053
time: 1.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0378069, upper bound: 0.0378053
time: 1.82 seconds

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

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 229

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 199

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0378424, upper bound: 0.0378480
time: 2.08 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0378448, upper bound: 0.0378458
time: 1.84 seconds

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

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 72

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0374118, upper bound: 0.0374092
time: 2.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0374118, upper bound: 0.0374092
time: 2.52 seconds

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

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 107

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0380156, upper bound: 0.0380285
time: 1.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0380280, upper bound: 0.0380162
time: 2.08 seconds

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

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 188

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 156

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 88

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0379645, upper bound: 0.0379684
time: 2.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0379675, upper bound: 0.0379656
time: 1.64 seconds

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

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 156

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 134

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0386148, upper bound: 0.0386146
time: 2.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0386148, upper bound: 0.0386147
time: 3.07 seconds

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

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 229

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 212

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0386163, upper bound: 0.0386161
time: 1.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0386161, upper bound: 0.0386163
time: 2.15 seconds

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

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 229

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0365609, upper bound: 0.0365605
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0365609, upper bound: 0.0365605
time: 1.03 seconds

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

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 118

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0387618, upper bound: 0.0387693
time: 1.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0387648, upper bound: 0.0387674
time: 2.18 seconds

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

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 112

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0388536, upper bound: 0.0388492
time: 1.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0388488, upper bound: 0.0388535
time: 1.95 seconds

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

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0388509, upper bound: 0.0388467
time: 2.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0388467, upper bound: 0.0388511
time: 1.98 seconds

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

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 72

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0371610, upper bound: 0.0371653
time: 1.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0371610, upper bound: 0.0371653
time: 1.63 seconds

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

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 229

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0385538, upper bound: 0.0385440
time: 2.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0385514, upper bound: 0.0385468
time: 2.17 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 5.23 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -0.0385543, upper bound: 0.0385731
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -0.0385551, upper bound: 0.0385651
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -0.0385527, upper bound: 0.0385713
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -0.0385523, upper bound: 0.0385712
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -0.0385621, upper bound: 0.0385761
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -0.0385621, upper bound: 0.0385762
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -0.0385438, upper bound: 0.0385653
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -0.0385439, upper bound: 0.0385653
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -0.0391097, upper bound: 0.0391381
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -0.0391098, upper bound: 0.0391381
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -0.0398025, upper bound: 0.0398182
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -0.0398088, upper bound: 0.0398177
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 5.23
Output dim: 3, lower bound: -0.0365935, upper bound: 0.0365990
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 5.23
Output dim: 3, lower bound: -0.0365935, upper bound: 0.0365990
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -0.0384712, upper bound: 0.0384932
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -0.0384754, upper bound: 0.0384841
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -0.0390021, upper bound: 0.0389944
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -0.0390021, upper bound: 0.0389944
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -0.0398638, upper bound: 0.0398700
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -0.0398777, upper bound: 0.0398733
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -0.0399267, upper bound: 0.0399147
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -0.0399074, upper bound: 0.0399190
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -0.0381030, upper bound: 0.0380989
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -0.0381030, upper bound: 0.0380989
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -0.0387532, upper bound: 0.0387215
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -0.0387532, upper bound: 0.0387215
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -0.0388636, upper bound: 0.0388353
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -0.0388583, upper bound: 0.0388404
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -0.0377250, upper bound: 0.0377302
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -0.0377271, upper bound: 0.0377282
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -0.0379153, upper bound: 0.0379117
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -0.0379152, upper bound: 0.0379117
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -0.0378832, upper bound: 0.0378881
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -0.0378832, upper bound: 0.0378881
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -0.0378070, upper bound: 0.0378053
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -0.0378069, upper bound: 0.0378053
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -0.0378424, upper bound: 0.0378480
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -0.0378448, upper bound: 0.0378458
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 5.23
Output dim: 3, lower bound: -0.0374118, upper bound: 0.0374092
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 5.23
Output dim: 3, lower bound: -0.0374118, upper bound: 0.0374092
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -0.0380156, upper bound: 0.0380285
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -0.0380280, upper bound: 0.0380162
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -0.0379645, upper bound: 0.0379684
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -0.0379675, upper bound: 0.0379656
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -0.0386148, upper bound: 0.0386146
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -0.0386148, upper bound: 0.0386147
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -0.0386163, upper bound: 0.0386161
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -0.0386161, upper bound: 0.0386163
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 5.23
Output dim: 3, lower bound: -0.0365609, upper bound: 0.0365605
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 5.23
Output dim: 3, lower bound: -0.0365609, upper bound: 0.0365605
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -0.0387618, upper bound: 0.0387693
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -0.0387648, upper bound: 0.0387674
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -0.0388536, upper bound: 0.0388492
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -0.0388488, upper bound: 0.0388535
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -0.0388509, upper bound: 0.0388467
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -0.0388467, upper bound: 0.0388511
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 5.23
Output dim: 3, lower bound: -0.0371610, upper bound: 0.0371653
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 5.23
Output dim: 3, lower bound: -0.0371610, upper bound: 0.0371653
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -0.0385538, upper bound: 0.0385440
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -0.0385514, upper bound: 0.0385468

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

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 156

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0366184, upper bound: 0.0366254
time: 1.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0366184, upper bound: 0.0366254
time: 1.68 seconds

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

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 212

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0385537, upper bound: 0.0385639
time: 2.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0385535, upper bound: 0.0385639
time: 2.52 seconds

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

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 78

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0384375, upper bound: 0.0384562
time: 2.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0384375, upper bound: 0.0384562
time: 1.96 seconds

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

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 156

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 134

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0385496, upper bound: 0.0385685
time: 2.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0385493, upper bound: 0.0385687
time: 2.08 seconds

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

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 134

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0385594, upper bound: 0.0385734
time: 1.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0385591, upper bound: 0.0385736
time: 1.89 seconds

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

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 156

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0366557, upper bound: 0.0366619
time: 1.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0366557, upper bound: 0.0366619
time: 1.50 seconds

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

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 61

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0384713, upper bound: 0.0384927
time: 1.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0384713, upper bound: 0.0384927
time: 2.02 seconds

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

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 156

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 175

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0384893, upper bound: 0.0385108
time: 1.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0384893, upper bound: 0.0385108
time: 1.88 seconds

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

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 156

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0370489, upper bound: 0.0370560
time: 1.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0370489, upper bound: 0.0370560
time: 1.69 seconds

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

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 229

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0379598, upper bound: 0.0379728
time: 1.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0379598, upper bound: 0.0379728
time: 1.77 seconds

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

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 188

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0391068, upper bound: 0.0391212
time: 2.13 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0391068, upper bound: 0.0391212
time: 2.35 seconds

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

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 108

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 212

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0398010, upper bound: 0.0398162
time: 3.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0398010, upper bound: 0.0398162
time: 2.71 seconds

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

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0365423, upper bound: 0.0365490
time: 1.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0365423, upper bound: 0.0365490
time: 1.31 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 72

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 199

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0382799, upper bound: 0.0382903
time: 2.00 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0382816, upper bound: 0.0382881
time: 2.07 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 72

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0389991, upper bound: 0.0389862
time: 1.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0389946, upper bound: 0.0389916
time: 1.94 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 207

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 72

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0387156, upper bound: 0.0387063
time: 2.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0387156, upper bound: 0.0387063
time: 1.82 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 156

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0383001, upper bound: 0.0382911
time: 2.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0383001, upper bound: 0.0382911
time: 2.19 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0381045, upper bound: 0.0381035
time: 1.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0381045, upper bound: 0.0381035
time: 1.83 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0387566, upper bound: 0.0387454
time: 2.05 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0387566, upper bound: 0.0387454
time: 2.05 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 200

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0384820, upper bound: 0.0384849
time: 2.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0384820, upper bound: 0.0384849
time: 2.42 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0376440, upper bound: 0.0376394
time: 1.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0376440, upper bound: 0.0376394
time: 1.87 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 207

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0380450, upper bound: 0.0380457
time: 1.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0380502, upper bound: 0.0380421
time: 1.75 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 200

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 229

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0387502, upper bound: 0.0387134
time: 2.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0387454, upper bound: 0.0387181
time: 2.32 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0387483, upper bound: 0.0387151
time: 1.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0387452, upper bound: 0.0387167
time: 2.36 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 78

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0387450, upper bound: 0.0387163
time: 2.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0387451, upper bound: 0.0387165
time: 3.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 78

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0387130, upper bound: 0.0387215
time: 2.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0387400, upper bound: 0.0387216
time: 3.16 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0377250, upper bound: 0.0377276
time: 1.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0377220, upper bound: 0.0377302
time: 1.88 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 212

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0377262, upper bound: 0.0377274
time: 1.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0377261, upper bound: 0.0377274
time: 1.96 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 229

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 107

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0376151, upper bound: 0.0376186
time: 1.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0376241, upper bound: 0.0376129
time: 1.98 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 156

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 200

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0359722, upper bound: 0.0359724
time: 1.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0359722, upper bound: 0.0359724
time: 1.19 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 207

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 229

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0378804, upper bound: 0.0378837
time: 1.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0378784, upper bound: 0.0378853
time: 1.73 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 188

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 156

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0378283, upper bound: 0.0378363
time: 2.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0378313, upper bound: 0.0378322
time: 1.99 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0377524, upper bound: 0.0377542
time: 2.11 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0377559, upper bound: 0.0377515
time: 1.82 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0378030, upper bound: 0.0377977
time: 9.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0377998, upper bound: 0.0378014
time: 1.93 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Candidate
type: DSZ, layer: 1, pos: 61

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0377773, upper bound: 0.0377825
time: 2.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0377770, upper bound: 0.0377826
time: 2.17 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 229

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0378420, upper bound: 0.0378409
time: 2.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0378406, upper bound: 0.0378430
time: 1.79 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 118

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0379226, upper bound: 0.0379357
time: 2.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0379226, upper bound: 0.0379321
time: 2.33 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 112

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0380272, upper bound: 0.0380127
time: 1.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0380235, upper bound: 0.0380153
time: 2.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0379080, upper bound: 0.0379163
time: 2.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0379082, upper bound: 0.0379120
time: 2.09 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 188

### Candidate
type: DSZ, layer: 1, pos: 107

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0376670, upper bound: 0.0376721
time: 1.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0376753, upper bound: 0.0376662
time: 2.04 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0360926, upper bound: 0.0360924
time: 1.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0360926, upper bound: 0.0360924
time: 1.13 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 72

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 88

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0383242, upper bound: 0.0383326
time: 2.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0383329, upper bound: 0.0383236
time: 3.57 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 88

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 61

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0385567, upper bound: 0.0385566
time: 2.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0385568, upper bound: 0.0385565
time: 2.35 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0372927, upper bound: 0.0372929
time: 1.90 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0372927, upper bound: 0.0372929
time: 1.94 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 1.01 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 3.98 + 596.40 = 600.38 seconds
