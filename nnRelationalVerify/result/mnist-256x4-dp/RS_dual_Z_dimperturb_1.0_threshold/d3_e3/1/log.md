## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 2.411746211


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716)
1: (-0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653)
2: (-0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282)
3: (-1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487)
4: (-1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829)
5: (-1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234)
6: (-1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365)
7: (-1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017)
8: (-1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066)
9: (-1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.47 + 3.82 = 5.29 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -2.4863377, upper bound: 2.4863377

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4856967, upper bound: 2.4856560
time: 2.44 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4856560, upper bound: 2.4856967
time: 2.74 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 5.32 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 5.32
Output dim: 8, lower bound: -2.4856967, upper bound: 2.4856560
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 5.32
Output dim: 8, lower bound: -2.4856560, upper bound: 2.4856967

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4852067, upper bound: 2.4854399
time: 2.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4855021, upper bound: 2.4851476
time: 1.70 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4851477, upper bound: 2.4855021
time: 2.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4854399, upper bound: 2.4852067
time: 1.99 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 5.91 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 5.91
Output dim: 8, lower bound: -2.4852067, upper bound: 2.4854399
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 5.91
Output dim: 8, lower bound: -2.4855021, upper bound: 2.4851476
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 5.91
Output dim: 8, lower bound: -2.4851477, upper bound: 2.4855021
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 5.91
Output dim: 8, lower bound: -2.4854399, upper bound: 2.4852067

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4700677, upper bound: 2.4704271
time: 1.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4700677, upper bound: 2.4704271
time: 1.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4706121, upper bound: 2.4698686
time: 1.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4706121, upper bound: 2.4698686
time: 1.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4698686, upper bound: 2.4706121
time: 1.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4698686, upper bound: 2.4706121
time: 1.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4704271, upper bound: 2.4700677
time: 1.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4704271, upper bound: 2.4700677
time: 1.68 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 4.76 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.76
Output dim: 8, lower bound: -2.4700677, upper bound: 2.4704271
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.76
Output dim: 8, lower bound: -2.4700677, upper bound: 2.4704271
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.76
Output dim: 8, lower bound: -2.4706121, upper bound: 2.4698686
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.76
Output dim: 8, lower bound: -2.4706121, upper bound: 2.4698686
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.76
Output dim: 8, lower bound: -2.4698686, upper bound: 2.4706121
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.76
Output dim: 8, lower bound: -2.4698686, upper bound: 2.4706121
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.76
Output dim: 8, lower bound: -2.4704271, upper bound: 2.4700677
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.76
Output dim: 8, lower bound: -2.4704271, upper bound: 2.4700677

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4522728, upper bound: 2.4526385
time: 1.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4522743, upper bound: 2.4526385
time: 1.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4522728, upper bound: 2.4526385
time: 1.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4522743, upper bound: 2.4526385
time: 1.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4528365, upper bound: 2.4521248
time: 1.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4528354, upper bound: 2.4521231
time: 2.03 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4528365, upper bound: 2.4521248
time: 1.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4528354, upper bound: 2.4521231
time: 1.99 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4521231, upper bound: 2.4528354
time: 1.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4521248, upper bound: 2.4528365
time: 1.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4521231, upper bound: 2.4528354
time: 1.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4521248, upper bound: 2.4528365
time: 1.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4526385, upper bound: 2.4522743
time: 1.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4526385, upper bound: 2.4522728
time: 1.49 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4526385, upper bound: 2.4522743
time: 1.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4526385, upper bound: 2.4522728
time: 1.48 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 4.41 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.41
Output dim: 8, lower bound: -2.4522728, upper bound: 2.4526385
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.41
Output dim: 8, lower bound: -2.4522743, upper bound: 2.4526385
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.41
Output dim: 8, lower bound: -2.4522728, upper bound: 2.4526385
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.41
Output dim: 8, lower bound: -2.4522743, upper bound: 2.4526385
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.41
Output dim: 8, lower bound: -2.4528365, upper bound: 2.4521248
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.41
Output dim: 8, lower bound: -2.4528354, upper bound: 2.4521231
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.41
Output dim: 8, lower bound: -2.4528365, upper bound: 2.4521248
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.41
Output dim: 8, lower bound: -2.4528354, upper bound: 2.4521231
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.41
Output dim: 8, lower bound: -2.4521231, upper bound: 2.4528354
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.41
Output dim: 8, lower bound: -2.4521248, upper bound: 2.4528365
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.41
Output dim: 8, lower bound: -2.4521231, upper bound: 2.4528354
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.41
Output dim: 8, lower bound: -2.4521248, upper bound: 2.4528365
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.41
Output dim: 8, lower bound: -2.4526385, upper bound: 2.4522743
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.41
Output dim: 8, lower bound: -2.4526385, upper bound: 2.4522728
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.41
Output dim: 8, lower bound: -2.4526385, upper bound: 2.4522743
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.41
Output dim: 8, lower bound: -2.4526385, upper bound: 2.4522728

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4521119, upper bound: 2.4524748
time: 1.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4521119, upper bound: 2.4524748
time: 1.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4521128, upper bound: 2.4524747
time: 1.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4521128, upper bound: 2.4524747
time: 1.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4521119, upper bound: 2.4524748
time: 1.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4521119, upper bound: 2.4524748
time: 1.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4521128, upper bound: 2.4524747
time: 1.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4521128, upper bound: 2.4524747
time: 1.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4526634, upper bound: 2.4519637
time: 1.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4526634, upper bound: 2.4519637
time: 1.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4526626, upper bound: 2.4519637
time: 1.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4526626, upper bound: 2.4519637
time: 1.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4526634, upper bound: 2.4519637
time: 1.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4526634, upper bound: 2.4519637
time: 2.01 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4526626, upper bound: 2.4519637
time: 1.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4526626, upper bound: 2.4519637
time: 1.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519637, upper bound: 2.4526626
time: 1.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519637, upper bound: 2.4526626
time: 1.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519637, upper bound: 2.4526634
time: 1.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519637, upper bound: 2.4526634
time: 1.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519637, upper bound: 2.4526626
time: 1.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519637, upper bound: 2.4526626
time: 1.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519637, upper bound: 2.4526634
time: 1.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519637, upper bound: 2.4526634
time: 1.54 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4524747, upper bound: 2.4521128
time: 1.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4524747, upper bound: 2.4521128
time: 1.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4524748, upper bound: 2.4521119
time: 1.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4524748, upper bound: 2.4521119
time: 1.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4524747, upper bound: 2.4521128
time: 1.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4524747, upper bound: 2.4521128
time: 1.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4524748, upper bound: 2.4521119
time: 1.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4524748, upper bound: 2.4521119
time: 1.65 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 4.72 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 8, lower bound: -2.4521119, upper bound: 2.4524748
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 8, lower bound: -2.4521119, upper bound: 2.4524748
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 8, lower bound: -2.4521128, upper bound: 2.4524747
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 8, lower bound: -2.4521128, upper bound: 2.4524747
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 8, lower bound: -2.4521119, upper bound: 2.4524748
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 8, lower bound: -2.4521119, upper bound: 2.4524748
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 8, lower bound: -2.4521128, upper bound: 2.4524747
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 8, lower bound: -2.4521128, upper bound: 2.4524747
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 8, lower bound: -2.4526634, upper bound: 2.4519637
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 8, lower bound: -2.4526634, upper bound: 2.4519637
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 8, lower bound: -2.4526626, upper bound: 2.4519637
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 8, lower bound: -2.4526626, upper bound: 2.4519637
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 8, lower bound: -2.4526634, upper bound: 2.4519637
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 8, lower bound: -2.4526634, upper bound: 2.4519637
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 8, lower bound: -2.4526626, upper bound: 2.4519637
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 8, lower bound: -2.4526626, upper bound: 2.4519637
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 8, lower bound: -2.4519637, upper bound: 2.4526626
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 8, lower bound: -2.4519637, upper bound: 2.4526626
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 8, lower bound: -2.4519637, upper bound: 2.4526634
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 8, lower bound: -2.4519637, upper bound: 2.4526634
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 8, lower bound: -2.4519637, upper bound: 2.4526626
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 8, lower bound: -2.4519637, upper bound: 2.4526626
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 8, lower bound: -2.4519637, upper bound: 2.4526634
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 8, lower bound: -2.4519637, upper bound: 2.4526634
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 8, lower bound: -2.4524747, upper bound: 2.4521128
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 8, lower bound: -2.4524747, upper bound: 2.4521128
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 8, lower bound: -2.4524748, upper bound: 2.4521119
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 8, lower bound: -2.4524748, upper bound: 2.4521119
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 8, lower bound: -2.4524747, upper bound: 2.4521128
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 8, lower bound: -2.4524747, upper bound: 2.4521128
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 8, lower bound: -2.4524748, upper bound: 2.4521119
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 8, lower bound: -2.4524748, upper bound: 2.4521119

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4521119, upper bound: 2.4524742
time: 1.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4521040, upper bound: 2.4524748
time: 1.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4521119, upper bound: 2.4524742
time: 1.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4521040, upper bound: 2.4524748
time: 1.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4521128, upper bound: 2.4524740
time: 1.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4521044, upper bound: 2.4524747
time: 1.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4521128, upper bound: 2.4524740
time: 1.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4521044, upper bound: 2.4524747
time: 1.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4521119, upper bound: 2.4524742
time: 1.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4521040, upper bound: 2.4524748
time: 1.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4521119, upper bound: 2.4524742
time: 1.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4521040, upper bound: 2.4524748
time: 1.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4521128, upper bound: 2.4524740
time: 1.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4521044, upper bound: 2.4524747
time: 1.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4521128, upper bound: 2.4524740
time: 1.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4521044, upper bound: 2.4524747
time: 2.06 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4526634, upper bound: 2.4519617
time: 1.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4526534, upper bound: 2.4519637
time: 1.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4526634, upper bound: 2.4519617
time: 1.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4526534, upper bound: 2.4519637
time: 1.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4526626, upper bound: 2.4519616
time: 1.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4526534, upper bound: 2.4519637
time: 1.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4526626, upper bound: 2.4519616
time: 1.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4526534, upper bound: 2.4519637
time: 1.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4526634, upper bound: 2.4519617
time: 1.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4526534, upper bound: 2.4519637
time: 1.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4526634, upper bound: 2.4519617
time: 1.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4526534, upper bound: 2.4519637
time: 1.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4526626, upper bound: 2.4519616
time: 1.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4526534, upper bound: 2.4519637
time: 1.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4526626, upper bound: 2.4519616
time: 1.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4526534, upper bound: 2.4519637
time: 1.50 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519637, upper bound: 2.4526534
time: 1.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519616, upper bound: 2.4526626
time: 1.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519637, upper bound: 2.4526534
time: 1.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519616, upper bound: 2.4526626
time: 1.54 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519637, upper bound: 2.4526534
time: 1.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519617, upper bound: 2.4526634
time: 1.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519637, upper bound: 2.4526534
time: 1.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519617, upper bound: 2.4526634
time: 1.51 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519637, upper bound: 2.4526534
time: 1.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519616, upper bound: 2.4526626
time: 1.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519637, upper bound: 2.4526534
time: 1.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519616, upper bound: 2.4526626
time: 1.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519637, upper bound: 2.4526534
time: 1.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519617, upper bound: 2.4526634
time: 1.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519637, upper bound: 2.4526534
time: 1.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519617, upper bound: 2.4526634
time: 1.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4524747, upper bound: 2.4521044
time: 1.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4524740, upper bound: 2.4521128
time: 1.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4524747, upper bound: 2.4521044
time: 1.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4524740, upper bound: 2.4521128
time: 1.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4524748, upper bound: 2.4521040
time: 1.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4524742, upper bound: 2.4521119
time: 1.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4524748, upper bound: 2.4521040
time: 1.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4524742, upper bound: 2.4521119
time: 1.94 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4524747, upper bound: 2.4521044
time: 1.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4524740, upper bound: 2.4521128
time: 1.83 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4524747, upper bound: 2.4521044
time: 1.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4524740, upper bound: 2.4521128
time: 1.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4524748, upper bound: 2.4521040
time: 1.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4524742, upper bound: 2.4521119
time: 1.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4524748, upper bound: 2.4521040
time: 1.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4524742, upper bound: 2.4521119
time: 1.65 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 5.06 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 8, lower bound: -2.4521119, upper bound: 2.4524742
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 8, lower bound: -2.4521040, upper bound: 2.4524748
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 8, lower bound: -2.4521119, upper bound: 2.4524742
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 8, lower bound: -2.4521040, upper bound: 2.4524748
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 8, lower bound: -2.4521128, upper bound: 2.4524740
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 8, lower bound: -2.4521044, upper bound: 2.4524747
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 8, lower bound: -2.4521128, upper bound: 2.4524740
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 8, lower bound: -2.4521044, upper bound: 2.4524747
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 8, lower bound: -2.4521119, upper bound: 2.4524742
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 8, lower bound: -2.4521040, upper bound: 2.4524748
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 8, lower bound: -2.4521119, upper bound: 2.4524742
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 8, lower bound: -2.4521040, upper bound: 2.4524748
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 8, lower bound: -2.4521128, upper bound: 2.4524740
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 8, lower bound: -2.4521044, upper bound: 2.4524747
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 8, lower bound: -2.4521128, upper bound: 2.4524740
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 8, lower bound: -2.4521044, upper bound: 2.4524747
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 8, lower bound: -2.4526634, upper bound: 2.4519617
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 8, lower bound: -2.4526534, upper bound: 2.4519637
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 8, lower bound: -2.4526634, upper bound: 2.4519617
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 8, lower bound: -2.4526534, upper bound: 2.4519637
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 8, lower bound: -2.4526626, upper bound: 2.4519616
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 8, lower bound: -2.4526534, upper bound: 2.4519637
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 8, lower bound: -2.4526626, upper bound: 2.4519616
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 8, lower bound: -2.4526534, upper bound: 2.4519637
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 8, lower bound: -2.4526634, upper bound: 2.4519617
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 8, lower bound: -2.4526534, upper bound: 2.4519637
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 8, lower bound: -2.4526634, upper bound: 2.4519617
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 8, lower bound: -2.4526534, upper bound: 2.4519637
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 8, lower bound: -2.4526626, upper bound: 2.4519616
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 8, lower bound: -2.4526534, upper bound: 2.4519637
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 8, lower bound: -2.4526626, upper bound: 2.4519616
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 8, lower bound: -2.4526534, upper bound: 2.4519637
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 8, lower bound: -2.4519637, upper bound: 2.4526534
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 8, lower bound: -2.4519616, upper bound: 2.4526626
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 8, lower bound: -2.4519637, upper bound: 2.4526534
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 8, lower bound: -2.4519616, upper bound: 2.4526626
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 8, lower bound: -2.4519637, upper bound: 2.4526534
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 8, lower bound: -2.4519617, upper bound: 2.4526634
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 8, lower bound: -2.4519637, upper bound: 2.4526534
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 8, lower bound: -2.4519617, upper bound: 2.4526634
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 8, lower bound: -2.4519637, upper bound: 2.4526534
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 8, lower bound: -2.4519616, upper bound: 2.4526626
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 8, lower bound: -2.4519637, upper bound: 2.4526534
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 8, lower bound: -2.4519616, upper bound: 2.4526626
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 8, lower bound: -2.4519637, upper bound: 2.4526534
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 8, lower bound: -2.4519617, upper bound: 2.4526634
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 8, lower bound: -2.4519637, upper bound: 2.4526534
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 8, lower bound: -2.4519617, upper bound: 2.4526634
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 8, lower bound: -2.4524747, upper bound: 2.4521044
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 8, lower bound: -2.4524740, upper bound: 2.4521128
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 8, lower bound: -2.4524747, upper bound: 2.4521044
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 8, lower bound: -2.4524740, upper bound: 2.4521128
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 8, lower bound: -2.4524748, upper bound: 2.4521040
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 8, lower bound: -2.4524742, upper bound: 2.4521119
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 8, lower bound: -2.4524748, upper bound: 2.4521040
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 8, lower bound: -2.4524742, upper bound: 2.4521119
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 8, lower bound: -2.4524747, upper bound: 2.4521044
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 8, lower bound: -2.4524740, upper bound: 2.4521128
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 8, lower bound: -2.4524747, upper bound: 2.4521044
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 8, lower bound: -2.4524740, upper bound: 2.4521128
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 8, lower bound: -2.4524748, upper bound: 2.4521040
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 8, lower bound: -2.4524742, upper bound: 2.4521119
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 8, lower bound: -2.4524748, upper bound: 2.4521040
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.06
Output dim: 8, lower bound: -2.4524742, upper bound: 2.4521119

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4520003, upper bound: 2.4523428
time: 1.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519470, upper bound: 2.4523542
time: 1.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519922, upper bound: 2.4523451
time: 1.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519375, upper bound: 2.4523547
time: 1.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4520003, upper bound: 2.4523428
time: 1.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519470, upper bound: 2.4523542
time: 1.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519922, upper bound: 2.4523451
time: 1.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519375, upper bound: 2.4523547
time: 1.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4520013, upper bound: 2.4523423
time: 1.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519474, upper bound: 2.4523541
time: 1.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519924, upper bound: 2.4523444
time: 1.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519379, upper bound: 2.4523547
time: 1.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4520013, upper bound: 2.4523423
time: 1.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519474, upper bound: 2.4523542
time: 1.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519924, upper bound: 2.4523444
time: 1.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519379, upper bound: 2.4523547
time: 1.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4520003, upper bound: 2.4523428
time: 1.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519467, upper bound: 2.4523542
time: 1.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519922, upper bound: 2.4523451
time: 3.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519374, upper bound: 2.4523547
time: 1.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4520003, upper bound: 2.4523428
time: 1.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519467, upper bound: 2.4523542
time: 1.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519922, upper bound: 2.4523451
time: 1.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519374, upper bound: 2.4523547
time: 1.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4520013, upper bound: 2.4523423
time: 1.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519473, upper bound: 2.4523541
time: 1.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519924, upper bound: 2.4523444
time: 1.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519378, upper bound: 2.4523547
time: 1.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4520013, upper bound: 2.4523423
time: 1.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519473, upper bound: 2.4523542
time: 1.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519924, upper bound: 2.4523444
time: 1.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4519378, upper bound: 2.4523547
time: 1.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4525455, upper bound: 2.4518077
time: 1.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4525230, upper bound: 2.4518457
time: 1.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4525357, upper bound: 2.4518108
time: 1.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4525127, upper bound: 2.4518473
time: 1.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4525455, upper bound: 2.4518077
time: 1.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4525230, upper bound: 2.4518457
time: 1.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4525357, upper bound: 2.4518108
time: 1.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4525127, upper bound: 2.4518473
time: 1.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4525448, upper bound: 2.4518045
time: 1.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4525231, upper bound: 2.4518453
time: 1.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4525357, upper bound: 2.4518087
time: 1.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4525129, upper bound: 2.4518466
time: 1.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4525448, upper bound: 2.4518045
time: 1.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4525231, upper bound: 2.4518453
time: 1.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4525357, upper bound: 2.4518087
time: 1.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4525129, upper bound: 2.4518466
time: 1.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4525455, upper bound: 2.4518077
time: 1.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4525230, upper bound: 2.4518457
time: 1.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4525357, upper bound: 2.4518108
time: 1.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4525127, upper bound: 2.4518473
time: 1.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4525455, upper bound: 2.4518077
time: 1.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4525230, upper bound: 2.4518457
time: 1.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4525357, upper bound: 2.4518108
time: 1.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4525127, upper bound: 2.4518473
time: 1.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4525448, upper bound: 2.4518045
time: 1.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4525231, upper bound: 2.4518453
time: 2.09 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4525357, upper bound: 2.4518087
time: 1.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4525129, upper bound: 2.4518466
time: 1.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4525448, upper bound: 2.4518045
time: 1.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4525231, upper bound: 2.4518453
time: 1.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4525357, upper bound: 2.4518087
time: 1.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4525129, upper bound: 2.4518466
time: 1.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4518466, upper bound: 2.4525129
time: 1.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4518087, upper bound: 2.4525357
time: 1.85 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4518453, upper bound: 2.4525231
time: 1.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4518045, upper bound: 2.4525448
time: 1.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4518466, upper bound: 2.4525129
time: 1.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4518087, upper bound: 2.4525357
time: 1.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4518453, upper bound: 2.4525231
time: 1.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4518045, upper bound: 2.4525448
time: 1.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4518473, upper bound: 2.4525127
time: 1.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4518108, upper bound: 2.4525357
time: 1.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4518457, upper bound: 2.4525230
time: 1.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4518077, upper bound: 2.4525455
time: 1.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4518473, upper bound: 2.4525127
time: 1.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4518108, upper bound: 2.4525357
time: 1.92 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4518457, upper bound: 2.4525230
time: 1.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4518077, upper bound: 2.4525455
time: 1.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4518466, upper bound: 2.4525129
time: 1.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4518087, upper bound: 2.4525357
time: 1.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4518453, upper bound: 2.4525231
time: 1.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4518045, upper bound: 2.4525448
time: 1.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4518466, upper bound: 2.4525129
time: 1.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4518087, upper bound: 2.4525357
time: 1.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4518453, upper bound: 2.4525231
time: 1.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4518045, upper bound: 2.4525448
time: 1.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4518473, upper bound: 2.4525127
time: 1.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4518108, upper bound: 2.4525357
time: 1.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4518457, upper bound: 2.4525230
time: 1.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4518077, upper bound: 2.4525455
time: 1.54 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4518473, upper bound: 2.4525127
time: 2.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4518108, upper bound: 2.4525357
time: 1.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4518457, upper bound: 2.4525230
time: 1.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4518077, upper bound: 2.4525455
time: 1.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4523547, upper bound: 2.4519378
time: 1.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4523444, upper bound: 2.4519924
time: 1.99 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4523542, upper bound: 2.4519473
time: 1.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4523423, upper bound: 2.4520013
time: 1.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4523547, upper bound: 2.4519378
time: 1.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4523444, upper bound: 2.4519924
time: 1.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4523542, upper bound: 2.4519473
time: 1.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4523423, upper bound: 2.4520013
time: 1.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4523547, upper bound: 2.4519374
time: 1.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4523451, upper bound: 2.4519922
time: 1.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4523542, upper bound: 2.4519467
time: 1.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4523428, upper bound: 2.4520003
time: 1.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4523547, upper bound: 2.4519374
time: 1.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4523451, upper bound: 2.4519922
time: 1.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4523542, upper bound: 2.4519467
time: 1.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4523428, upper bound: 2.4520003
time: 2.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.1449089, 1.0237626, -1.1449089, 1.0237626, -2.1686716, 2.1686716
1: -0.9533735, 0.9367918, -0.9533735, 0.9367918, -1.8901653, 1.8901653
2: -0.9885503, 1.1024778, -0.9885503, 1.1024778, -2.0910282, 2.0910282
3: -1.1323893, 1.0104594, -1.1323893, 1.0104594, -2.1428487, 2.1428487
4: -1.2352548, 0.9672282, -1.2352548, 0.9672282, -2.2024829, 2.2024829
5: -1.0193650, 0.9940586, -1.0193650, 0.9940586, -2.0134234, 2.0134234
6: -1.0222397, 1.0629967, -1.0222397, 1.0629967, -2.0852365, 2.0852365
7: -1.0907527, 1.1420490, -1.0907527, 1.1420490, -2.2328017, 2.2328017
8: -1.3188362, 1.3299705, -1.3188362, 1.3299705, -2.6488066, 2.6488066
9: -1.0492049, 1.1355765, -1.0492049, 1.1355765, -2.1847816, 2.1847816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=72, inp2_unstable=72, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=133, inp2_unstable=133, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4523547, upper bound: 2.4519379
time: 1.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.4523444, upper bound: 2.4519924
time: 1.88 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 5.19 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4520003, upper bound: 2.4523428
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4519470, upper bound: 2.4523542
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4519922, upper bound: 2.4523451
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4519375, upper bound: 2.4523547
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4520003, upper bound: 2.4523428
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4519470, upper bound: 2.4523542
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4519922, upper bound: 2.4523451
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4519375, upper bound: 2.4523547
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4520013, upper bound: 2.4523423
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4519474, upper bound: 2.4523541
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4519924, upper bound: 2.4523444
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4519379, upper bound: 2.4523547
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4520013, upper bound: 2.4523423
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4519474, upper bound: 2.4523542
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4519924, upper bound: 2.4523444
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4519379, upper bound: 2.4523547
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4520003, upper bound: 2.4523428
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4519467, upper bound: 2.4523542
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4519922, upper bound: 2.4523451
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4519374, upper bound: 2.4523547
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4520003, upper bound: 2.4523428
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4519467, upper bound: 2.4523542
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4519922, upper bound: 2.4523451
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4519374, upper bound: 2.4523547
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4520013, upper bound: 2.4523423
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4519473, upper bound: 2.4523541
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4519924, upper bound: 2.4523444
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4519378, upper bound: 2.4523547
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4520013, upper bound: 2.4523423
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4519473, upper bound: 2.4523542
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4519924, upper bound: 2.4523444
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4519378, upper bound: 2.4523547
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4525455, upper bound: 2.4518077
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4525230, upper bound: 2.4518457
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4525357, upper bound: 2.4518108
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4525127, upper bound: 2.4518473
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4525455, upper bound: 2.4518077
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4525230, upper bound: 2.4518457
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4525357, upper bound: 2.4518108
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4525127, upper bound: 2.4518473
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4525448, upper bound: 2.4518045
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4525231, upper bound: 2.4518453
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4525357, upper bound: 2.4518087
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4525129, upper bound: 2.4518466
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4525448, upper bound: 2.4518045
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4525231, upper bound: 2.4518453
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4525357, upper bound: 2.4518087
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4525129, upper bound: 2.4518466
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4525455, upper bound: 2.4518077
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4525230, upper bound: 2.4518457
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4525357, upper bound: 2.4518108
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4525127, upper bound: 2.4518473
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4525455, upper bound: 2.4518077
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4525230, upper bound: 2.4518457
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4525357, upper bound: 2.4518108
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4525127, upper bound: 2.4518473
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4525448, upper bound: 2.4518045
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4525231, upper bound: 2.4518453
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4525357, upper bound: 2.4518087
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4525129, upper bound: 2.4518466
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4525448, upper bound: 2.4518045
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4525231, upper bound: 2.4518453
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4525357, upper bound: 2.4518087
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4525129, upper bound: 2.4518466
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4518466, upper bound: 2.4525129
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4518087, upper bound: 2.4525357
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4518453, upper bound: 2.4525231
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4518045, upper bound: 2.4525448
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4518466, upper bound: 2.4525129
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4518087, upper bound: 2.4525357
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4518453, upper bound: 2.4525231
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4518045, upper bound: 2.4525448
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4518473, upper bound: 2.4525127
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4518108, upper bound: 2.4525357
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4518457, upper bound: 2.4525230
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4518077, upper bound: 2.4525455
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4518473, upper bound: 2.4525127
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4518108, upper bound: 2.4525357
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4518457, upper bound: 2.4525230
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4518077, upper bound: 2.4525455
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4518466, upper bound: 2.4525129
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4518087, upper bound: 2.4525357
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4518453, upper bound: 2.4525231
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4518045, upper bound: 2.4525448
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4518466, upper bound: 2.4525129
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4518087, upper bound: 2.4525357
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4518453, upper bound: 2.4525231
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4518045, upper bound: 2.4525448
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4518473, upper bound: 2.4525127
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4518108, upper bound: 2.4525357
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4518457, upper bound: 2.4525230
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4518077, upper bound: 2.4525455
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4518473, upper bound: 2.4525127
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4518108, upper bound: 2.4525357
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4518457, upper bound: 2.4525230
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4518077, upper bound: 2.4525455
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4523547, upper bound: 2.4519378
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4523444, upper bound: 2.4519924
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4523542, upper bound: 2.4519473
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4523423, upper bound: 2.4520013
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4523547, upper bound: 2.4519378
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4523444, upper bound: 2.4519924
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4523542, upper bound: 2.4519473
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4523423, upper bound: 2.4520013
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4523547, upper bound: 2.4519374
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4523451, upper bound: 2.4519922
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4523542, upper bound: 2.4519467
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4523428, upper bound: 2.4520003
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4523547, upper bound: 2.4519374
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4523451, upper bound: 2.4519922
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4523542, upper bound: 2.4519467
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4523428, upper bound: 2.4520003
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4523547, upper bound: 2.4519379
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.19
Output dim: 8, lower bound: -2.4523444, upper bound: 2.4519924
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.19
Output dim: 8, lower bound: -2.4524740, upper bound: 2.4521128
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.19
Output dim: 8, lower bound: -2.4524747, upper bound: 2.4521044
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.19
Output dim: 8, lower bound: -2.4524740, upper bound: 2.4521128
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.19
Output dim: 8, lower bound: -2.4524748, upper bound: 2.4521040
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.19
Output dim: 8, lower bound: -2.4524742, upper bound: 2.4521119
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.19
Output dim: 8, lower bound: -2.4524748, upper bound: 2.4521040
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.19
Output dim: 8, lower bound: -2.4524742, upper bound: 2.4521119

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 5.29 + 595.78 = 601.07 seconds
