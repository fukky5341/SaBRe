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
execution time: IAR + RelationalAnalysis = 1.19 + 3.09 = 4.28 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.0408667, upper bound: 0.0408667

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 229

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0406497, upper bound: 0.0406497
time: 2.73 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0406497, upper bound: 0.0406497
time: 2.55 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 5.29 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 5.29
Output dim: 3, lower bound: -0.0406497, upper bound: 0.0406497
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 5.29
Output dim: 3, lower bound: -0.0406497, upper bound: 0.0406497

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

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0391232, upper bound: 0.0391232
time: 1.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0391232, upper bound: 0.0391232
time: 1.94 seconds

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

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0404668, upper bound: 0.0404713
time: 2.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0404668, upper bound: 0.0404668
time: 2.40 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 6.26 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 6.26
Output dim: 3, lower bound: -0.0391232, upper bound: 0.0391232
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 6.26
Output dim: 3, lower bound: -0.0391232, upper bound: 0.0391232
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 6.26
Output dim: 3, lower bound: -0.0404668, upper bound: 0.0404713
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 6.26
Output dim: 3, lower bound: -0.0404668, upper bound: 0.0404668

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

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0391193, upper bound: 0.0391159
time: 1.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0391156, upper bound: 0.0391193
time: 1.96 seconds

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
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 156

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0391205, upper bound: 0.0391204
time: 1.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0391204, upper bound: 0.0391205
time: 1.68 seconds

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
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0403913, upper bound: 0.0404034
time: 3.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0403913, upper bound: 0.0403943
time: 2.44 seconds

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

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0404325, upper bound: 0.0404325
time: 2.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0404325, upper bound: 0.0404366
time: 2.57 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 5.90 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 5.90
Output dim: 3, lower bound: -0.0391193, upper bound: 0.0391159
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 5.90
Output dim: 3, lower bound: -0.0391156, upper bound: 0.0391193
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 5.90
Output dim: 3, lower bound: -0.0391205, upper bound: 0.0391204
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 5.90
Output dim: 3, lower bound: -0.0391204, upper bound: 0.0391205
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 5.90
Output dim: 3, lower bound: -0.0403913, upper bound: 0.0404034
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 5.90
Output dim: 3, lower bound: -0.0403913, upper bound: 0.0403943
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 5.90
Output dim: 3, lower bound: -0.0404325, upper bound: 0.0404325
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 5.90
Output dim: 3, lower bound: -0.0404325, upper bound: 0.0404366

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

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0391060, upper bound: 0.0391005
time: 2.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0391025, upper bound: 0.0391027
time: 1.78 seconds

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
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0390239, upper bound: 0.0390304
time: 1.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0390269, upper bound: 0.0390287
time: 1.68 seconds

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

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 156

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 200

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 57

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0373908, upper bound: 0.0373910
time: 2.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0373908, upper bound: 0.0373910
time: 2.01 seconds

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

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 207

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0379007, upper bound: 0.0379006
time: 1.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0379007, upper bound: 0.0379006
time: 1.57 seconds

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
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0403606, upper bound: 0.0403680
time: 2.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0403559, upper bound: 0.0403726
time: 2.46 seconds

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

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 108

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0403717, upper bound: 0.0403668
time: 2.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0403633, upper bound: 0.0403670
time: 4.12 seconds

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

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 57

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392921, upper bound: 0.0392884
time: 2.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392884, upper bound: 0.0392884
time: 2.22 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0404336, upper bound: 0.0404336
time: 2.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0404296, upper bound: 0.0404339
time: 3.03 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 6.48 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.48
Output dim: 3, lower bound: -0.0391060, upper bound: 0.0391005
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.48
Output dim: 3, lower bound: -0.0391025, upper bound: 0.0391027
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.48
Output dim: 3, lower bound: -0.0390239, upper bound: 0.0390304
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.48
Output dim: 3, lower bound: -0.0390269, upper bound: 0.0390287
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 6.48
Output dim: 3, lower bound: -0.0373908, upper bound: 0.0373910
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 6.48
Output dim: 3, lower bound: -0.0373908, upper bound: 0.0373910
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.48
Output dim: 3, lower bound: -0.0379007, upper bound: 0.0379006
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.48
Output dim: 3, lower bound: -0.0379007, upper bound: 0.0379006
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.48
Output dim: 3, lower bound: -0.0403606, upper bound: 0.0403680
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.48
Output dim: 3, lower bound: -0.0403559, upper bound: 0.0403726
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.48
Output dim: 3, lower bound: -0.0403717, upper bound: 0.0403668
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.48
Output dim: 3, lower bound: -0.0403633, upper bound: 0.0403670
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.48
Output dim: 3, lower bound: -0.0392921, upper bound: 0.0392884
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.48
Output dim: 3, lower bound: -0.0392884, upper bound: 0.0392884
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.48
Output dim: 3, lower bound: -0.0404336, upper bound: 0.0404336
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.48
Output dim: 3, lower bound: -0.0404296, upper bound: 0.0404339

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

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 156

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0389500, upper bound: 0.0389473
time: 1.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0389530, upper bound: 0.0389435
time: 1.84 seconds

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

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0389713, upper bound: 0.0389711
time: 2.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0389713, upper bound: 0.0389713
time: 2.35 seconds

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
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 57

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 107

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0386889, upper bound: 0.0387028
time: 2.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0386957, upper bound: 0.0386928
time: 2.93 seconds

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

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 156

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 200

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0386853, upper bound: 0.0386905
time: 1.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0386888, upper bound: 0.0386880
time: 2.60 seconds

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

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 207

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 156

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0378098, upper bound: 0.0378102
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0378103, upper bound: 0.0378097
time: 1.12 seconds

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
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 156

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0378365, upper bound: 0.0378390
time: 1.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0378390, upper bound: 0.0378359
time: 1.36 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 72

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0403121, upper bound: 0.0403305
time: 2.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0403071, upper bound: 0.0403184
time: 2.32 seconds

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
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0403524, upper bound: 0.0403698
time: 2.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0403524, upper bound: 0.0403699
time: 4.26 seconds

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

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0403521, upper bound: 0.0403564
time: 2.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0403622, upper bound: 0.0403634
time: 3.32 seconds

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

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 207

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 188

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0398271, upper bound: 0.0398236
time: 2.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0398271, upper bound: 0.0398236
time: 2.48 seconds

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
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 200

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0391729, upper bound: 0.0391729
time: 3.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0391758, upper bound: 0.0391729
time: 2.45 seconds

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

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0391785, upper bound: 0.0391729
time: 2.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0391758, upper bound: 0.0391729
time: 2.44 seconds

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

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0404252, upper bound: 0.0404270
time: 2.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0404252, upper bound: 0.0404326
time: 2.56 seconds

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

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 207

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0403708, upper bound: 0.0403751
time: 2.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0403750, upper bound: 0.0403751
time: 2.56 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 5.94 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.94
Output dim: 3, lower bound: -0.0389500, upper bound: 0.0389473
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.94
Output dim: 3, lower bound: -0.0389530, upper bound: 0.0389435
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.94
Output dim: 3, lower bound: -0.0389713, upper bound: 0.0389711
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.94
Output dim: 3, lower bound: -0.0389713, upper bound: 0.0389713
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.94
Output dim: 3, lower bound: -0.0386889, upper bound: 0.0387028
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.94
Output dim: 3, lower bound: -0.0386957, upper bound: 0.0386928
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.94
Output dim: 3, lower bound: -0.0386853, upper bound: 0.0386905
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.94
Output dim: 3, lower bound: -0.0386888, upper bound: 0.0386880
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.94
Output dim: 3, lower bound: -0.0378098, upper bound: 0.0378102
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.94
Output dim: 3, lower bound: -0.0378103, upper bound: 0.0378097
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.94
Output dim: 3, lower bound: -0.0378365, upper bound: 0.0378390
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.94
Output dim: 3, lower bound: -0.0378390, upper bound: 0.0378359
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.94
Output dim: 3, lower bound: -0.0403121, upper bound: 0.0403305
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.94
Output dim: 3, lower bound: -0.0403071, upper bound: 0.0403184
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.94
Output dim: 3, lower bound: -0.0403524, upper bound: 0.0403698
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.94
Output dim: 3, lower bound: -0.0403524, upper bound: 0.0403699
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.94
Output dim: 3, lower bound: -0.0403521, upper bound: 0.0403564
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.94
Output dim: 3, lower bound: -0.0403622, upper bound: 0.0403634
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.94
Output dim: 3, lower bound: -0.0398271, upper bound: 0.0398236
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.94
Output dim: 3, lower bound: -0.0398271, upper bound: 0.0398236
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.94
Output dim: 3, lower bound: -0.0391729, upper bound: 0.0391729
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.94
Output dim: 3, lower bound: -0.0391758, upper bound: 0.0391729
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.94
Output dim: 3, lower bound: -0.0391785, upper bound: 0.0391729
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.94
Output dim: 3, lower bound: -0.0391758, upper bound: 0.0391729
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.94
Output dim: 3, lower bound: -0.0404252, upper bound: 0.0404270
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.94
Output dim: 3, lower bound: -0.0404252, upper bound: 0.0404326
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.94
Output dim: 3, lower bound: -0.0403708, upper bound: 0.0403751
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.94
Output dim: 3, lower bound: -0.0403750, upper bound: 0.0403751

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

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0377427, upper bound: 0.0377392
time: 1.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0377427, upper bound: 0.0377392
time: 1.86 seconds

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

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 175

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0388758, upper bound: 0.0388660
time: 1.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0388758, upper bound: 0.0388662
time: 1.77 seconds

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

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0388800, upper bound: 0.0388833
time: 1.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0388833, upper bound: 0.0388799
time: 2.03 seconds

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

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0389686, upper bound: 0.0389685
time: 1.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0389682, upper bound: 0.0389686
time: 1.90 seconds

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

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0386880, upper bound: 0.0386987
time: 2.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0386860, upper bound: 0.0387020
time: 2.01 seconds

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

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0386305, upper bound: 0.0386272
time: 1.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0386305, upper bound: 0.0386275
time: 1.64 seconds

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

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0386823, upper bound: 0.0386878
time: 1.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0386824, upper bound: 0.0386879
time: 1.92 seconds

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

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 229

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 107

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0383605, upper bound: 0.0383668
time: 2.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0383586, upper bound: 0.0383618
time: 1.94 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0374147, upper bound: 0.0374147
time: 1.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0374147, upper bound: 0.0374147
time: 1.75 seconds

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
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 207

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0376545, upper bound: 0.0376556
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0376562, upper bound: 0.0376546
time: 1.45 seconds

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 118

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 212

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0378354, upper bound: 0.0378381
time: 1.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0378353, upper bound: 0.0378382
time: 1.78 seconds

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 207

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 108

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0378091, upper bound: 0.0378058
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0378088, upper bound: 0.0378060
time: 2.06 seconds

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0402477, upper bound: 0.0402713
time: 2.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0402526, upper bound: 0.0402713
time: 3.46 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 212

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0403055, upper bound: 0.0403168
time: 2.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0403055, upper bound: 0.0403166
time: 2.63 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 229

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0390717, upper bound: 0.0390854
time: 1.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0390717, upper bound: 0.0390854
time: 1.91 seconds

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

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 207

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0400090, upper bound: 0.0400261
time: 2.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0400090, upper bound: 0.0400261
time: 2.56 seconds

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 188

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0398089, upper bound: 0.0398135
time: 2.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0398089, upper bound: 0.0398135
time: 3.13 seconds

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

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0390565, upper bound: 0.0390588
time: 2.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0390565, upper bound: 0.0390588
time: 2.25 seconds

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

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 57

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0382156, upper bound: 0.0382148
time: 2.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0382156, upper bound: 0.0382148
time: 2.18 seconds

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

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 207

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0380367, upper bound: 0.0380369
time: 1.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0380367, upper bound: 0.0380369
time: 1.47 seconds

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

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 88

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 207

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0389529, upper bound: 0.0389464
time: 2.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0389529, upper bound: 0.0389464
time: 2.19 seconds

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

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 72

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 188

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0381730, upper bound: 0.0381680
time: 2.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0381730, upper bound: 0.0381680
time: 1.96 seconds

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
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0391749, upper bound: 0.0391648
time: 2.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0391698, upper bound: 0.0391694
time: 2.49 seconds

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0381447, upper bound: 0.0381401
time: 2.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0381447, upper bound: 0.0381401
time: 2.71 seconds

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 156

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0390371, upper bound: 0.0390336
time: 2.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0390371, upper bound: 0.0390336
time: 2.24 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 156

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 207

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0400815, upper bound: 0.0400895
time: 2.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0400815, upper bound: 0.0400895
time: 2.51 seconds

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
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0400536, upper bound: 0.0400579
time: 2.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0400536, upper bound: 0.0400579
time: 2.47 seconds

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0401129, upper bound: 0.0401330
time: 2.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0401129, upper bound: 0.0401177
time: 2.89 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 6.53 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.53
Output dim: 3, lower bound: -0.0377427, upper bound: 0.0377392
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.53
Output dim: 3, lower bound: -0.0377427, upper bound: 0.0377392
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.53
Output dim: 3, lower bound: -0.0388758, upper bound: 0.0388660
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.53
Output dim: 3, lower bound: -0.0388758, upper bound: 0.0388662
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.53
Output dim: 3, lower bound: -0.0388800, upper bound: 0.0388833
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.53
Output dim: 3, lower bound: -0.0388833, upper bound: 0.0388799
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.53
Output dim: 3, lower bound: -0.0389686, upper bound: 0.0389685
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.53
Output dim: 3, lower bound: -0.0389682, upper bound: 0.0389686
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.53
Output dim: 3, lower bound: -0.0386880, upper bound: 0.0386987
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.53
Output dim: 3, lower bound: -0.0386860, upper bound: 0.0387020
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.53
Output dim: 3, lower bound: -0.0386305, upper bound: 0.0386272
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.53
Output dim: 3, lower bound: -0.0386305, upper bound: 0.0386275
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.53
Output dim: 3, lower bound: -0.0386823, upper bound: 0.0386878
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.53
Output dim: 3, lower bound: -0.0386824, upper bound: 0.0386879
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.53
Output dim: 3, lower bound: -0.0383605, upper bound: 0.0383668
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.53
Output dim: 3, lower bound: -0.0383586, upper bound: 0.0383618
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 6.53
Output dim: 3, lower bound: -0.0374147, upper bound: 0.0374147
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 6.53
Output dim: 3, lower bound: -0.0374147, upper bound: 0.0374147
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.53
Output dim: 3, lower bound: -0.0376545, upper bound: 0.0376556
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.53
Output dim: 3, lower bound: -0.0376562, upper bound: 0.0376546
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.53
Output dim: 3, lower bound: -0.0378354, upper bound: 0.0378381
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.53
Output dim: 3, lower bound: -0.0378353, upper bound: 0.0378382
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.53
Output dim: 3, lower bound: -0.0378091, upper bound: 0.0378058
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.53
Output dim: 3, lower bound: -0.0378088, upper bound: 0.0378060
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.53
Output dim: 3, lower bound: -0.0402477, upper bound: 0.0402713
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.53
Output dim: 3, lower bound: -0.0402526, upper bound: 0.0402713
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.53
Output dim: 3, lower bound: -0.0403055, upper bound: 0.0403168
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.53
Output dim: 3, lower bound: -0.0403055, upper bound: 0.0403166
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.53
Output dim: 3, lower bound: -0.0390717, upper bound: 0.0390854
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.53
Output dim: 3, lower bound: -0.0390717, upper bound: 0.0390854
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.53
Output dim: 3, lower bound: -0.0400090, upper bound: 0.0400261
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.53
Output dim: 3, lower bound: -0.0400090, upper bound: 0.0400261
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.53
Output dim: 3, lower bound: -0.0398089, upper bound: 0.0398135
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.53
Output dim: 3, lower bound: -0.0398089, upper bound: 0.0398135
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.53
Output dim: 3, lower bound: -0.0390565, upper bound: 0.0390588
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.53
Output dim: 3, lower bound: -0.0390565, upper bound: 0.0390588
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.53
Output dim: 3, lower bound: -0.0382156, upper bound: 0.0382148
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.53
Output dim: 3, lower bound: -0.0382156, upper bound: 0.0382148
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.53
Output dim: 3, lower bound: -0.0380367, upper bound: 0.0380369
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.53
Output dim: 3, lower bound: -0.0380367, upper bound: 0.0380369
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.53
Output dim: 3, lower bound: -0.0389529, upper bound: 0.0389464
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.53
Output dim: 3, lower bound: -0.0389529, upper bound: 0.0389464
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.53
Output dim: 3, lower bound: -0.0381730, upper bound: 0.0381680
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.53
Output dim: 3, lower bound: -0.0381730, upper bound: 0.0381680
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.53
Output dim: 3, lower bound: -0.0391749, upper bound: 0.0391648
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.53
Output dim: 3, lower bound: -0.0391698, upper bound: 0.0391694
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.53
Output dim: 3, lower bound: -0.0381447, upper bound: 0.0381401
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.53
Output dim: 3, lower bound: -0.0381447, upper bound: 0.0381401
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.53
Output dim: 3, lower bound: -0.0390371, upper bound: 0.0390336
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.53
Output dim: 3, lower bound: -0.0390371, upper bound: 0.0390336
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.53
Output dim: 3, lower bound: -0.0400815, upper bound: 0.0400895
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.53
Output dim: 3, lower bound: -0.0400815, upper bound: 0.0400895
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.53
Output dim: 3, lower bound: -0.0400536, upper bound: 0.0400579
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.53
Output dim: 3, lower bound: -0.0400536, upper bound: 0.0400579
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.53
Output dim: 3, lower bound: -0.0401129, upper bound: 0.0401330
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.53
Output dim: 3, lower bound: -0.0401129, upper bound: 0.0401177

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 156

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 108

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0377120, upper bound: 0.0377085
time: 1.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0377113, upper bound: 0.0377086
time: 1.36 seconds

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

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0377403, upper bound: 0.0377365
time: 1.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0377396, upper bound: 0.0377368
time: 1.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

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

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 200

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 156

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0388750, upper bound: 0.0388638
time: 1.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0388718, upper bound: 0.0388651
time: 1.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

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

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 72

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 188

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0380611, upper bound: 0.0380553
time: 1.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0380611, upper bound: 0.0380553
time: 1.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

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

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 72

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0385531, upper bound: 0.0385589
time: 3.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0385570, upper bound: 0.0385538
time: 2.28 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

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

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 88

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 57

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 212

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0388742, upper bound: 0.0388782
time: 1.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0388742, upper bound: 0.0388782
time: 2.04 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 188

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0381755, upper bound: 0.0381749
time: 1.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0381755, upper bound: 0.0381749
time: 1.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

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

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 118

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 108

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0389393, upper bound: 0.0389397
time: 1.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0389389, upper bound: 0.0389396
time: 1.94 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

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

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 156

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 188

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0378167, upper bound: 0.0378235
time: 1.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0378167, upper bound: 0.0378235
time: 1.43 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 175

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0386130, upper bound: 0.0386287
time: 2.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0386126, upper bound: 0.0386287
time: 2.04 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

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

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 200

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0386257, upper bound: 0.0386189
time: 2.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0386218, upper bound: 0.0386222
time: 2.54 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 57

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0386296, upper bound: 0.0386240
time: 2.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0386268, upper bound: 0.0386266
time: 2.44 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

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

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 107

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0383557, upper bound: 0.0383682
time: 2.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0383617, upper bound: 0.0383611
time: 1.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 107

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0383556, upper bound: 0.0383682
time: 2.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0383619, upper bound: 0.0383612
time: 1.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 72

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 207

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0373404, upper bound: 0.0373451
time: 1.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0373404, upper bound: 0.0373451
time: 1.52 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

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

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 207

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0373471, upper bound: 0.0373421
time: 2.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0373471, upper bound: 0.0373421
time: 2.17 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0372184, upper bound: 0.0372194
time: 1.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0372184, upper bound: 0.0372194
time: 1.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

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

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 108

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0376254, upper bound: 0.0376232
time: 1.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0376252, upper bound: 0.0376238
time: 1.47 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

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

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 229

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 207

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0378346, upper bound: 0.0378359
time: 1.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0378319, upper bound: 0.0378375
time: 1.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 72

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0374029, upper bound: 0.0374039
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0374029, upper bound: 0.0374039
time: 1.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

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

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 229

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0377438, upper bound: 0.0377405
time: 1.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0377439, upper bound: 0.0377405
time: 1.22 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 156

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0378081, upper bound: 0.0378031
time: 1.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0378058, upper bound: 0.0378053
time: 1.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 88

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 175

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0401638, upper bound: 0.0401874
time: 2.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0401638, upper bound: 0.0401874
time: 2.27 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

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

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 156

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0388396, upper bound: 0.0388543
time: 2.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0388396, upper bound: 0.0388543
time: 2.84 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0390034, upper bound: 0.0390006
time: 2.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0390034, upper bound: 0.0390006
time: 2.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

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

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0400048, upper bound: 0.0400006
time: 2.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0399897, upper bound: 0.0400006
time: 7.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0390681, upper bound: 0.0390761
time: 7.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0390605, upper bound: 0.0390818
time: 1.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0390286, upper bound: 0.0390498
time: 2.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0390361, upper bound: 0.0390410
time: 1.93 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 57

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0389698, upper bound: 0.0389830
time: 2.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0389698, upper bound: 0.0389830
time: 2.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

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

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0398043, upper bound: 0.0398205
time: 2.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0398043, upper bound: 0.0398205
time: 2.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 212

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0398227, upper bound: 0.0398120
time: 2.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0398072, upper bound: 0.0398116
time: 2.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0397920, upper bound: 0.0397772
time: 2.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0397723, upper bound: 0.0397812
time: 2.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

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

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 57

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0364729, upper bound: 0.0364745
time: 1.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0364729, upper bound: 0.0364745
time: 1.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0390129, upper bound: 0.0390230
time: 2.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0390207, upper bound: 0.0390150
time: 2.18 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

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

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 175

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0381524, upper bound: 0.0381514
time: 1.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0381521, upper bound: 0.0381517
time: 2.19 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

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

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0381491, upper bound: 0.0381546
time: 2.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0381554, upper bound: 0.0381484
time: 2.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 212

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0380338, upper bound: 0.0380337
time: 1.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0380322, upper bound: 0.0380337
time: 2.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

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

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 107

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0376425, upper bound: 0.0376484
time: 1.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0376486, upper bound: 0.0376442
time: 1.93 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 229

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0389496, upper bound: 0.0389391
time: 2.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0389466, upper bound: 0.0389430
time: 2.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 156

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0363163, upper bound: 0.0363147
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0363163, upper bound: 0.0363147
time: 1.26 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 156

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 107

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0377626, upper bound: 0.0377693
time: 1.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0377743, upper bound: 0.0377587
time: 1.90 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0378158, upper bound: 0.0378161
time: 1.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0378204, upper bound: 0.0378109
time: 5.21 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

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

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 200

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0376002, upper bound: 0.0375952
time: 2.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0376002, upper bound: 0.0375952
time: 2.08 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

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

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 229
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0364932, upper bound: 0.0364941
time: 1.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0364932, upper bound: 0.0364941
time: 1.32 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 3.82 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 3, lower bound: -0.0377120, upper bound: 0.0377085
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 3, lower bound: -0.0377113, upper bound: 0.0377086
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 3, lower bound: -0.0377403, upper bound: 0.0377365
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 3, lower bound: -0.0377396, upper bound: 0.0377368
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 3, lower bound: -0.0388750, upper bound: 0.0388638
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 3, lower bound: -0.0388718, upper bound: 0.0388651
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 3, lower bound: -0.0380611, upper bound: 0.0380553
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 3, lower bound: -0.0380611, upper bound: 0.0380553
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 3, lower bound: -0.0385531, upper bound: 0.0385589
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 3, lower bound: -0.0385570, upper bound: 0.0385538
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 3, lower bound: -0.0388742, upper bound: 0.0388782
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 3, lower bound: -0.0388742, upper bound: 0.0388782
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 3, lower bound: -0.0381755, upper bound: 0.0381749
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 3, lower bound: -0.0381755, upper bound: 0.0381749
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 3, lower bound: -0.0389393, upper bound: 0.0389397
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 3, lower bound: -0.0389389, upper bound: 0.0389396
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 3, lower bound: -0.0378167, upper bound: 0.0378235
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 3, lower bound: -0.0378167, upper bound: 0.0378235
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 3, lower bound: -0.0386130, upper bound: 0.0386287
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 3, lower bound: -0.0386126, upper bound: 0.0386287
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 3, lower bound: -0.0386257, upper bound: 0.0386189
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 3, lower bound: -0.0386218, upper bound: 0.0386222
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 3, lower bound: -0.0386296, upper bound: 0.0386240
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 3, lower bound: -0.0386268, upper bound: 0.0386266
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 3, lower bound: -0.0383557, upper bound: 0.0383682
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 3, lower bound: -0.0383617, upper bound: 0.0383611
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 3, lower bound: -0.0383556, upper bound: 0.0383682
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 3, lower bound: -0.0383619, upper bound: 0.0383612
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.82
Output dim: 3, lower bound: -0.0373404, upper bound: 0.0373451
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.82
Output dim: 3, lower bound: -0.0373404, upper bound: 0.0373451
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.82
Output dim: 3, lower bound: -0.0373471, upper bound: 0.0373421
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.82
Output dim: 3, lower bound: -0.0373471, upper bound: 0.0373421
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.82
Output dim: 3, lower bound: -0.0372184, upper bound: 0.0372194
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.82
Output dim: 3, lower bound: -0.0372184, upper bound: 0.0372194
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 3, lower bound: -0.0376254, upper bound: 0.0376232
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 3, lower bound: -0.0376252, upper bound: 0.0376238
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 3, lower bound: -0.0378346, upper bound: 0.0378359
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 3, lower bound: -0.0378319, upper bound: 0.0378375
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.82
Output dim: 3, lower bound: -0.0374029, upper bound: 0.0374039
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.82
Output dim: 3, lower bound: -0.0374029, upper bound: 0.0374039
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 3, lower bound: -0.0377438, upper bound: 0.0377405
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 3, lower bound: -0.0377439, upper bound: 0.0377405
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 3, lower bound: -0.0378081, upper bound: 0.0378031
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 3, lower bound: -0.0378058, upper bound: 0.0378053
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 3, lower bound: -0.0401638, upper bound: 0.0401874
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 3, lower bound: -0.0401638, upper bound: 0.0401874
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 3, lower bound: -0.0388396, upper bound: 0.0388543
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 3, lower bound: -0.0388396, upper bound: 0.0388543
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 3, lower bound: -0.0390034, upper bound: 0.0390006
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 3, lower bound: -0.0390034, upper bound: 0.0390006
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 3, lower bound: -0.0400048, upper bound: 0.0400006
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 3, lower bound: -0.0399897, upper bound: 0.0400006
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 3, lower bound: -0.0390681, upper bound: 0.0390761
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 3, lower bound: -0.0390605, upper bound: 0.0390818
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 3, lower bound: -0.0390286, upper bound: 0.0390498
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 3, lower bound: -0.0390361, upper bound: 0.0390410
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 3, lower bound: -0.0389698, upper bound: 0.0389830
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 3, lower bound: -0.0389698, upper bound: 0.0389830
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 3, lower bound: -0.0398043, upper bound: 0.0398205
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 3, lower bound: -0.0398043, upper bound: 0.0398205
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 3, lower bound: -0.0398227, upper bound: 0.0398120
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 3, lower bound: -0.0398072, upper bound: 0.0398116
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 3, lower bound: -0.0397920, upper bound: 0.0397772
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 3, lower bound: -0.0397723, upper bound: 0.0397812
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.82
Output dim: 3, lower bound: -0.0364729, upper bound: 0.0364745
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.82
Output dim: 3, lower bound: -0.0364729, upper bound: 0.0364745
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 3, lower bound: -0.0390129, upper bound: 0.0390230
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 3, lower bound: -0.0390207, upper bound: 0.0390150
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 3, lower bound: -0.0381524, upper bound: 0.0381514
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 3, lower bound: -0.0381521, upper bound: 0.0381517
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 3, lower bound: -0.0381491, upper bound: 0.0381546
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 3, lower bound: -0.0381554, upper bound: 0.0381484
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 3, lower bound: -0.0380338, upper bound: 0.0380337
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 3, lower bound: -0.0380322, upper bound: 0.0380337
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 3, lower bound: -0.0376425, upper bound: 0.0376484
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 3, lower bound: -0.0376486, upper bound: 0.0376442
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 3, lower bound: -0.0389496, upper bound: 0.0389391
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 3, lower bound: -0.0389466, upper bound: 0.0389430
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.82
Output dim: 3, lower bound: -0.0363163, upper bound: 0.0363147
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.82
Output dim: 3, lower bound: -0.0363163, upper bound: 0.0363147
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 3, lower bound: -0.0377626, upper bound: 0.0377693
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 3, lower bound: -0.0377743, upper bound: 0.0377587
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 3, lower bound: -0.0378158, upper bound: 0.0378161
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 3, lower bound: -0.0378204, upper bound: 0.0378109
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 3, lower bound: -0.0376002, upper bound: 0.0375952
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 3, lower bound: -0.0376002, upper bound: 0.0375952
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.82
Output dim: 3, lower bound: -0.0364932, upper bound: 0.0364941
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.82
Output dim: 3, lower bound: -0.0364932, upper bound: 0.0364941
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 3, lower bound: -0.0381447, upper bound: 0.0381401
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 3, lower bound: -0.0381447, upper bound: 0.0381401
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 3, lower bound: -0.0390371, upper bound: 0.0390336
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 3, lower bound: -0.0390371, upper bound: 0.0390336
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 3, lower bound: -0.0400815, upper bound: 0.0400895
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 3, lower bound: -0.0400815, upper bound: 0.0400895
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 3, lower bound: -0.0400536, upper bound: 0.0400579
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 3, lower bound: -0.0400536, upper bound: 0.0400579
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 3, lower bound: -0.0401129, upper bound: 0.0401330
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 3, lower bound: -0.0401129, upper bound: 0.0401177

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 4.28 + 597.59 = 601.87 seconds
