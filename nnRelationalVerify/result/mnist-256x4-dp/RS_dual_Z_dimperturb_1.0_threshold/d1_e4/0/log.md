## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.037597364


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

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
execution time: IAR + RelationalAnalysis = 1.22 + 3.08 = 4.30 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.0408667, upper bound: 0.0408667

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 188

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0402928, upper bound: 0.0402928
time: 2.47 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0402928, upper bound: 0.0402928
time: 2.44 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 5.03 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 5.03
Output dim: 3, lower bound: -0.0402928, upper bound: 0.0402928
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 5.03
Output dim: 3, lower bound: -0.0402928, upper bound: 0.0402928

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0402347, upper bound: 0.0402456
time: 2.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0402456, upper bound: 0.0402347
time: 2.43 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0402347, upper bound: 0.0402456
time: 2.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0402456, upper bound: 0.0402347
time: 2.83 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 6.68 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 6.68
Output dim: 3, lower bound: -0.0402347, upper bound: 0.0402456
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 6.68
Output dim: 3, lower bound: -0.0402456, upper bound: 0.0402347
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 6.68
Output dim: 3, lower bound: -0.0402347, upper bound: 0.0402456
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 6.68
Output dim: 3, lower bound: -0.0402456, upper bound: 0.0402347

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 207

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0396954, upper bound: 0.0397060
time: 2.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0396954, upper bound: 0.0397060
time: 2.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 207

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0397060, upper bound: 0.0396954
time: 2.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0397060, upper bound: 0.0396954
time: 2.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 207

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0396954, upper bound: 0.0397060
time: 2.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0396954, upper bound: 0.0397060
time: 3.23 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 207

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0397060, upper bound: 0.0396954
time: 2.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0397060, upper bound: 0.0396954
time: 2.17 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 5.51 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 5.51
Output dim: 3, lower bound: -0.0396954, upper bound: 0.0397060
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 5.51
Output dim: 3, lower bound: -0.0396954, upper bound: 0.0397060
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 5.51
Output dim: 3, lower bound: -0.0397060, upper bound: 0.0396954
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 5.51
Output dim: 3, lower bound: -0.0397060, upper bound: 0.0396954
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 5.51
Output dim: 3, lower bound: -0.0396954, upper bound: 0.0397060
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 5.51
Output dim: 3, lower bound: -0.0396954, upper bound: 0.0397060
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 5.51
Output dim: 3, lower bound: -0.0397060, upper bound: 0.0396954
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 5.51
Output dim: 3, lower bound: -0.0397060, upper bound: 0.0396954

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0396679, upper bound: 0.0396755
time: 2.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0396641, upper bound: 0.0396785
time: 2.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0396679, upper bound: 0.0396755
time: 2.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0396641, upper bound: 0.0396786
time: 2.15 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0396785, upper bound: 0.0396641
time: 2.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0396755, upper bound: 0.0396679
time: 2.15 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0396641, upper bound: 0.0396641
time: 2.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0396755, upper bound: 0.0396679
time: 2.17 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0396641, upper bound: 0.0396755
time: 2.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0396641, upper bound: 0.0396786
time: 3.26 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0396679, upper bound: 0.0396755
time: 2.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0396641, upper bound: 0.0396786
time: 2.45 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0396785, upper bound: 0.0396641
time: 2.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0396755, upper bound: 0.0396679
time: 2.19 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0396785, upper bound: 0.0396641
time: 2.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0396755, upper bound: 0.0396679
time: 2.25 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 5.52 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.52
Output dim: 3, lower bound: -0.0396679, upper bound: 0.0396755
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.52
Output dim: 3, lower bound: -0.0396641, upper bound: 0.0396785
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.52
Output dim: 3, lower bound: -0.0396679, upper bound: 0.0396755
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.52
Output dim: 3, lower bound: -0.0396641, upper bound: 0.0396786
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.52
Output dim: 3, lower bound: -0.0396785, upper bound: 0.0396641
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.52
Output dim: 3, lower bound: -0.0396755, upper bound: 0.0396679
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.52
Output dim: 3, lower bound: -0.0396641, upper bound: 0.0396641
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.52
Output dim: 3, lower bound: -0.0396755, upper bound: 0.0396679
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.52
Output dim: 3, lower bound: -0.0396641, upper bound: 0.0396755
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.52
Output dim: 3, lower bound: -0.0396641, upper bound: 0.0396786
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.52
Output dim: 3, lower bound: -0.0396679, upper bound: 0.0396755
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.52
Output dim: 3, lower bound: -0.0396641, upper bound: 0.0396786
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.52
Output dim: 3, lower bound: -0.0396785, upper bound: 0.0396641
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.52
Output dim: 3, lower bound: -0.0396755, upper bound: 0.0396679
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.52
Output dim: 3, lower bound: -0.0396785, upper bound: 0.0396641
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.52
Output dim: 3, lower bound: -0.0396755, upper bound: 0.0396679

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393482, upper bound: 0.0393607
time: 2.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393521, upper bound: 0.0393604
time: 2.23 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393482, upper bound: 0.0393638
time: 2.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393482, upper bound: 0.0393636
time: 2.22 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393482, upper bound: 0.0393607
time: 3.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393521, upper bound: 0.0393604
time: 2.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393482, upper bound: 0.0393638
time: 2.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393482, upper bound: 0.0393636
time: 2.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393636, upper bound: 0.0393484
time: 2.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393638, upper bound: 0.0393482
time: 2.12 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393604, upper bound: 0.0393521
time: 2.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393482, upper bound: 0.0393521
time: 2.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393636, upper bound: 0.0393484
time: 2.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393638, upper bound: 0.0393482
time: 2.13 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393604, upper bound: 0.0393521
time: 2.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393482, upper bound: 0.0393521
time: 2.44 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393521, upper bound: 0.0393607
time: 2.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393521, upper bound: 0.0393604
time: 2.18 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393482, upper bound: 0.0393638
time: 2.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393482, upper bound: 0.0393636
time: 2.23 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393521, upper bound: 0.0393607
time: 2.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393521, upper bound: 0.0393604
time: 2.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393482, upper bound: 0.0393638
time: 2.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393482, upper bound: 0.0393636
time: 3.20 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393636, upper bound: 0.0393484
time: 2.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393638, upper bound: 0.0393482
time: 2.19 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393604, upper bound: 0.0393521
time: 2.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393482, upper bound: 0.0393521
time: 3.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393636, upper bound: 0.0393484
time: 2.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393638, upper bound: 0.0393482
time: 2.19 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393604, upper bound: 0.0393521
time: 2.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393482, upper bound: 0.0393521
time: 2.48 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 10.03 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 10.03
Output dim: 3, lower bound: -0.0393482, upper bound: 0.0393607
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 10.03
Output dim: 3, lower bound: -0.0393521, upper bound: 0.0393604
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 10.03
Output dim: 3, lower bound: -0.0393482, upper bound: 0.0393638
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 10.03
Output dim: 3, lower bound: -0.0393482, upper bound: 0.0393636
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 10.03
Output dim: 3, lower bound: -0.0393482, upper bound: 0.0393607
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 10.03
Output dim: 3, lower bound: -0.0393521, upper bound: 0.0393604
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 10.03
Output dim: 3, lower bound: -0.0393482, upper bound: 0.0393638
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 10.03
Output dim: 3, lower bound: -0.0393482, upper bound: 0.0393636
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 10.03
Output dim: 3, lower bound: -0.0393636, upper bound: 0.0393484
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 10.03
Output dim: 3, lower bound: -0.0393638, upper bound: 0.0393482
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 10.03
Output dim: 3, lower bound: -0.0393604, upper bound: 0.0393521
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 10.03
Output dim: 3, lower bound: -0.0393482, upper bound: 0.0393521
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 10.03
Output dim: 3, lower bound: -0.0393636, upper bound: 0.0393484
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 10.03
Output dim: 3, lower bound: -0.0393638, upper bound: 0.0393482
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 10.03
Output dim: 3, lower bound: -0.0393604, upper bound: 0.0393521
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 10.03
Output dim: 3, lower bound: -0.0393482, upper bound: 0.0393521
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 10.03
Output dim: 3, lower bound: -0.0393521, upper bound: 0.0393607
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 10.03
Output dim: 3, lower bound: -0.0393521, upper bound: 0.0393604
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 10.03
Output dim: 3, lower bound: -0.0393482, upper bound: 0.0393638
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 10.03
Output dim: 3, lower bound: -0.0393482, upper bound: 0.0393636
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 10.03
Output dim: 3, lower bound: -0.0393521, upper bound: 0.0393607
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 10.03
Output dim: 3, lower bound: -0.0393521, upper bound: 0.0393604
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 10.03
Output dim: 3, lower bound: -0.0393482, upper bound: 0.0393638
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 10.03
Output dim: 3, lower bound: -0.0393482, upper bound: 0.0393636
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 10.03
Output dim: 3, lower bound: -0.0393636, upper bound: 0.0393484
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 10.03
Output dim: 3, lower bound: -0.0393638, upper bound: 0.0393482
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 10.03
Output dim: 3, lower bound: -0.0393604, upper bound: 0.0393521
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 10.03
Output dim: 3, lower bound: -0.0393482, upper bound: 0.0393521
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 10.03
Output dim: 3, lower bound: -0.0393636, upper bound: 0.0393484
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 10.03
Output dim: 3, lower bound: -0.0393638, upper bound: 0.0393482
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 10.03
Output dim: 3, lower bound: -0.0393604, upper bound: 0.0393521
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 10.03
Output dim: 3, lower bound: -0.0393482, upper bound: 0.0393521

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392845, upper bound: 0.0392988
time: 5.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392902, upper bound: 0.0392901
time: 2.14 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392845, upper bound: 0.0392985
time: 2.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392903, upper bound: 0.0392898
time: 1.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392810, upper bound: 0.0393020
time: 22.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392864, upper bound: 0.0392932
time: 2.16 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392810, upper bound: 0.0393018
time: 2.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392810, upper bound: 0.0392932
time: 2.26 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392810, upper bound: 0.0392988
time: 2.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392902, upper bound: 0.0392900
time: 2.20 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392845, upper bound: 0.0392985
time: 2.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392903, upper bound: 0.0392898
time: 1.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392810, upper bound: 0.0393020
time: 2.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392810, upper bound: 0.0392932
time: 2.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392810, upper bound: 0.0393018
time: 2.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392810, upper bound: 0.0392932
time: 2.45 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392845, upper bound: 0.0392866
time: 2.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393018, upper bound: 0.0392811
time: 2.44 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392932, upper bound: 0.0392864
time: 2.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393020, upper bound: 0.0392810
time: 2.19 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392898, upper bound: 0.0392902
time: 2.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392985, upper bound: 0.0392845
time: 2.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392900, upper bound: 0.0392902
time: 2.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392989, upper bound: 0.0392845
time: 2.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392810, upper bound: 0.0392866
time: 2.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393018, upper bound: 0.0392811
time: 2.18 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392932, upper bound: 0.0392864
time: 2.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393020, upper bound: 0.0392810
time: 2.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392898, upper bound: 0.0392903
time: 2.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392985, upper bound: 0.0392845
time: 2.26 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392900, upper bound: 0.0392902
time: 2.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392989, upper bound: 0.0392845
time: 2.24 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392845, upper bound: 0.0392988
time: 2.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392902, upper bound: 0.0392901
time: 2.14 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392845, upper bound: 0.0392985
time: 2.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392903, upper bound: 0.0392898
time: 1.85 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392810, upper bound: 0.0393020
time: 2.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392864, upper bound: 0.0392933
time: 3.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392810, upper bound: 0.0393018
time: 2.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392810, upper bound: 0.0392932
time: 2.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392845, upper bound: 0.0392989
time: 2.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392902, upper bound: 0.0392900
time: 2.12 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392845, upper bound: 0.0392985
time: 2.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392903, upper bound: 0.0392898
time: 1.92 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392810, upper bound: 0.0393020
time: 2.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392864, upper bound: 0.0392932
time: 3.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392810, upper bound: 0.0393018
time: 2.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392810, upper bound: 0.0392932
time: 2.17 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392810, upper bound: 0.0392866
time: 2.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393018, upper bound: 0.0392810
time: 2.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392845, upper bound: 0.0392864
time: 2.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393020, upper bound: 0.0392810
time: 2.22 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392898, upper bound: 0.0392903
time: 2.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392985, upper bound: 0.0392846
time: 2.26 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392900, upper bound: 0.0392902
time: 2.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392989, upper bound: 0.0392845
time: 2.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392810, upper bound: 0.0392866
time: 2.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393018, upper bound: 0.0392811
time: 2.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392810, upper bound: 0.0392864
time: 2.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393020, upper bound: 0.0392810
time: 2.05 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392898, upper bound: 0.0392903
time: 2.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392985, upper bound: 0.0392846
time: 2.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392900, upper bound: 0.0392902
time: 2.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392989, upper bound: 0.0392845
time: 2.27 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 9.83 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.83
Output dim: 3, lower bound: -0.0392845, upper bound: 0.0392988
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.83
Output dim: 3, lower bound: -0.0392902, upper bound: 0.0392901
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.83
Output dim: 3, lower bound: -0.0392845, upper bound: 0.0392985
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.83
Output dim: 3, lower bound: -0.0392903, upper bound: 0.0392898
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.83
Output dim: 3, lower bound: -0.0392810, upper bound: 0.0393020
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.83
Output dim: 3, lower bound: -0.0392864, upper bound: 0.0392932
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.83
Output dim: 3, lower bound: -0.0392810, upper bound: 0.0393018
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.83
Output dim: 3, lower bound: -0.0392810, upper bound: 0.0392932
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.83
Output dim: 3, lower bound: -0.0392810, upper bound: 0.0392988
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.83
Output dim: 3, lower bound: -0.0392902, upper bound: 0.0392900
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.83
Output dim: 3, lower bound: -0.0392845, upper bound: 0.0392985
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.83
Output dim: 3, lower bound: -0.0392903, upper bound: 0.0392898
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.83
Output dim: 3, lower bound: -0.0392810, upper bound: 0.0393020
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.83
Output dim: 3, lower bound: -0.0392810, upper bound: 0.0392932
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.83
Output dim: 3, lower bound: -0.0392810, upper bound: 0.0393018
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.83
Output dim: 3, lower bound: -0.0392810, upper bound: 0.0392932
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.83
Output dim: 3, lower bound: -0.0392845, upper bound: 0.0392866
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.83
Output dim: 3, lower bound: -0.0393018, upper bound: 0.0392811
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.83
Output dim: 3, lower bound: -0.0392932, upper bound: 0.0392864
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.83
Output dim: 3, lower bound: -0.0393020, upper bound: 0.0392810
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.83
Output dim: 3, lower bound: -0.0392898, upper bound: 0.0392902
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.83
Output dim: 3, lower bound: -0.0392985, upper bound: 0.0392845
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.83
Output dim: 3, lower bound: -0.0392900, upper bound: 0.0392902
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.83
Output dim: 3, lower bound: -0.0392989, upper bound: 0.0392845
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.83
Output dim: 3, lower bound: -0.0392810, upper bound: 0.0392866
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.83
Output dim: 3, lower bound: -0.0393018, upper bound: 0.0392811
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.83
Output dim: 3, lower bound: -0.0392932, upper bound: 0.0392864
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.83
Output dim: 3, lower bound: -0.0393020, upper bound: 0.0392810
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.83
Output dim: 3, lower bound: -0.0392898, upper bound: 0.0392903
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.83
Output dim: 3, lower bound: -0.0392985, upper bound: 0.0392845
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.83
Output dim: 3, lower bound: -0.0392900, upper bound: 0.0392902
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.83
Output dim: 3, lower bound: -0.0392989, upper bound: 0.0392845
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.83
Output dim: 3, lower bound: -0.0392845, upper bound: 0.0392988
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.83
Output dim: 3, lower bound: -0.0392902, upper bound: 0.0392901
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.83
Output dim: 3, lower bound: -0.0392845, upper bound: 0.0392985
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.83
Output dim: 3, lower bound: -0.0392903, upper bound: 0.0392898
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.83
Output dim: 3, lower bound: -0.0392810, upper bound: 0.0393020
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.83
Output dim: 3, lower bound: -0.0392864, upper bound: 0.0392933
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.83
Output dim: 3, lower bound: -0.0392810, upper bound: 0.0393018
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.83
Output dim: 3, lower bound: -0.0392810, upper bound: 0.0392932
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.83
Output dim: 3, lower bound: -0.0392845, upper bound: 0.0392989
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.83
Output dim: 3, lower bound: -0.0392902, upper bound: 0.0392900
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.83
Output dim: 3, lower bound: -0.0392845, upper bound: 0.0392985
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.83
Output dim: 3, lower bound: -0.0392903, upper bound: 0.0392898
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.83
Output dim: 3, lower bound: -0.0392810, upper bound: 0.0393020
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.83
Output dim: 3, lower bound: -0.0392864, upper bound: 0.0392932
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.83
Output dim: 3, lower bound: -0.0392810, upper bound: 0.0393018
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.83
Output dim: 3, lower bound: -0.0392810, upper bound: 0.0392932
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.83
Output dim: 3, lower bound: -0.0392810, upper bound: 0.0392866
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.83
Output dim: 3, lower bound: -0.0393018, upper bound: 0.0392810
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.83
Output dim: 3, lower bound: -0.0392845, upper bound: 0.0392864
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.83
Output dim: 3, lower bound: -0.0393020, upper bound: 0.0392810
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.83
Output dim: 3, lower bound: -0.0392898, upper bound: 0.0392903
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.83
Output dim: 3, lower bound: -0.0392985, upper bound: 0.0392846
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.83
Output dim: 3, lower bound: -0.0392900, upper bound: 0.0392902
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.83
Output dim: 3, lower bound: -0.0392989, upper bound: 0.0392845
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.83
Output dim: 3, lower bound: -0.0392810, upper bound: 0.0392866
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.83
Output dim: 3, lower bound: -0.0393018, upper bound: 0.0392811
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.83
Output dim: 3, lower bound: -0.0392810, upper bound: 0.0392864
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.83
Output dim: 3, lower bound: -0.0393020, upper bound: 0.0392810
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.83
Output dim: 3, lower bound: -0.0392898, upper bound: 0.0392903
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.83
Output dim: 3, lower bound: -0.0392985, upper bound: 0.0392846
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.83
Output dim: 3, lower bound: -0.0392900, upper bound: 0.0392902
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.83
Output dim: 3, lower bound: -0.0392989, upper bound: 0.0392845

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 212

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392707, upper bound: 0.0392880
time: 3.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392743, upper bound: 0.0392880
time: 2.53 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.14 seconds

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 4.30 + 596.69 = 600.99 seconds
