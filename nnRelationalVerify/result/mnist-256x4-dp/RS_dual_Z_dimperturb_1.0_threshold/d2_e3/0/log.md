## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 12.3086572218


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=60, inp2_unstable=60, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=230, inp2_unstable=230, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-7.0235176, 5.7524662, -7.0235176, 5.7524662, -12.7759838, 12.7759819)
1: (-6.1864815, 5.1664095, -6.1864815, 5.1664095, -11.3528910, 11.3528910)
2: (-7.9090343, 5.1656170, -7.9090343, 5.1656170, -13.0746517, 13.0746517)
3: (-8.4604549, 4.2214842, -8.4604549, 4.2214842, -12.6819391, 12.6819391)
4: (-8.3454752, 5.7502418, -8.3454752, 5.7502418, -14.0957165, 14.0957165)
5: (-7.1336541, 5.6869812, -7.1336541, 5.6869812, -12.8206348, 12.8206348)
6: (-6.3902140, 6.4361529, -6.3902140, 6.4361529, -12.8263664, 12.8263664)
7: (-7.1255641, 6.9407606, -7.1255641, 6.9407606, -14.0663242, 14.0663242)
8: (-10.3838263, 4.4998660, -10.3838263, 4.4998660, -14.8836918, 14.8836918)
9: (-6.2737536, 6.3332019, -6.2737536, 6.3332019, -12.6069536, 12.6069536)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.37 + 5.28 = 6.65 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -12.3209782, upper bound: 12.3209782

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3209462, upper bound: 12.3209459
time: 4.76 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3209459, upper bound: 12.3209462
time: 5.22 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 10.14 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 10.14
Output dim: 8, lower bound: -12.3209462, upper bound: 12.3209459
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 10.14
Output dim: 8, lower bound: -12.3209459, upper bound: 12.3209462

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -7.0235176, 5.7524662, -7.0235176, 5.7524662, -12.7759838, 12.7759819
1: -6.1864815, 5.1664095, -6.1864815, 5.1664095, -11.3528910, 11.3528910
2: -7.9090343, 5.1656170, -7.9090343, 5.1656170, -13.0746517, 13.0746517
3: -8.4604549, 4.2214842, -8.4604549, 4.2214842, -12.6819391, 12.6819391
4: -8.3454752, 5.7502418, -8.3454752, 5.7502418, -14.0957165, 14.0957165
5: -7.1336541, 5.6869812, -7.1336541, 5.6869812, -12.8206348, 12.8206348
6: -6.3902140, 6.4361529, -6.3902140, 6.4361529, -12.8263664, 12.8263664
7: -7.1255641, 6.9407606, -7.1255641, 6.9407606, -14.0663242, 14.0663242
8: -10.3838263, 4.4998660, -10.3838263, 4.4998660, -14.8836918, 14.8836918
9: -6.2737536, 6.3332019, -6.2737536, 6.3332019, -12.6069536, 12.6069536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=60, inp2_unstable=60, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=230, inp2_unstable=230, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3206969, upper bound: 12.3206969
time: 4.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3206969, upper bound: 12.3206963
time: 3.19 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -7.0235176, 5.7524662, -7.0235176, 5.7524662, -12.7759838, 12.7759819
1: -6.1864815, 5.1664095, -6.1864815, 5.1664095, -11.3528910, 11.3528910
2: -7.9090343, 5.1656170, -7.9090343, 5.1656170, -13.0746517, 13.0746517
3: -8.4604549, 4.2214842, -8.4604549, 4.2214842, -12.6819391, 12.6819391
4: -8.3454752, 5.7502418, -8.3454752, 5.7502418, -14.0957165, 14.0957165
5: -7.1336541, 5.6869812, -7.1336541, 5.6869812, -12.8206348, 12.8206348
6: -6.3902140, 6.4361529, -6.3902140, 6.4361529, -12.8263664, 12.8263664
7: -7.1255641, 6.9407606, -7.1255641, 6.9407606, -14.0663242, 14.0663242
8: -10.3838263, 4.4998660, -10.3838263, 4.4998660, -14.8836918, 14.8836918
9: -6.2737536, 6.3332019, -6.2737536, 6.3332019, -12.6069536, 12.6069536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=60, inp2_unstable=60, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=230, inp2_unstable=230, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3206969, upper bound: 12.3206969
time: 4.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3206969, upper bound: 12.3206962
time: 4.82 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 10.30 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 10.30
Output dim: 8, lower bound: -12.3206969, upper bound: 12.3206969
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 10.30
Output dim: 8, lower bound: -12.3206969, upper bound: 12.3206963
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 10.30
Output dim: 8, lower bound: -12.3206969, upper bound: 12.3206969
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 10.30
Output dim: 8, lower bound: -12.3206969, upper bound: 12.3206962

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.0235176, 5.7524662, -7.0235176, 5.7524662, -12.7759838, 12.7759819
1: -6.1864815, 5.1664095, -6.1864815, 5.1664095, -11.3528910, 11.3528910
2: -7.9090343, 5.1656170, -7.9090343, 5.1656170, -13.0746517, 13.0746517
3: -8.4604549, 4.2214842, -8.4604549, 4.2214842, -12.6819391, 12.6819391
4: -8.3454752, 5.7502418, -8.3454752, 5.7502418, -14.0957165, 14.0957165
5: -7.1336541, 5.6869812, -7.1336541, 5.6869812, -12.8206348, 12.8206348
6: -6.3902140, 6.4361529, -6.3902140, 6.4361529, -12.8263664, 12.8263664
7: -7.1255641, 6.9407606, -7.1255641, 6.9407606, -14.0663242, 14.0663242
8: -10.3838263, 4.4998660, -10.3838263, 4.4998660, -14.8836918, 14.8836918
9: -6.2737536, 6.3332019, -6.2737536, 6.3332019, -12.6069536, 12.6069536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=60, inp2_unstable=60, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=230, inp2_unstable=230, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205418, upper bound: 12.3205409
time: 4.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205409, upper bound: 12.3205414
time: 3.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.0235176, 5.7524662, -7.0235176, 5.7524662, -12.7759838, 12.7759819
1: -6.1864815, 5.1664095, -6.1864815, 5.1664095, -11.3528910, 11.3528910
2: -7.9090343, 5.1656170, -7.9090343, 5.1656170, -13.0746517, 13.0746517
3: -8.4604549, 4.2214842, -8.4604549, 4.2214842, -12.6819391, 12.6819391
4: -8.3454752, 5.7502418, -8.3454752, 5.7502418, -14.0957165, 14.0957165
5: -7.1336541, 5.6869812, -7.1336541, 5.6869812, -12.8206348, 12.8206348
6: -6.3902140, 6.4361529, -6.3902140, 6.4361529, -12.8263664, 12.8263664
7: -7.1255641, 6.9407606, -7.1255641, 6.9407606, -14.0663242, 14.0663242
8: -10.3838263, 4.4998660, -10.3838263, 4.4998660, -14.8836918, 14.8836918
9: -6.2737536, 6.3332019, -6.2737536, 6.3332019, -12.6069536, 12.6069536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=60, inp2_unstable=60, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=230, inp2_unstable=230, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205418, upper bound: 12.3205410
time: 3.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205409, upper bound: 12.3205414
time: 3.51 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.0235176, 5.7524662, -7.0235176, 5.7524662, -12.7759838, 12.7759819
1: -6.1864815, 5.1664095, -6.1864815, 5.1664095, -11.3528910, 11.3528910
2: -7.9090343, 5.1656170, -7.9090343, 5.1656170, -13.0746517, 13.0746517
3: -8.4604549, 4.2214842, -8.4604549, 4.2214842, -12.6819391, 12.6819391
4: -8.3454752, 5.7502418, -8.3454752, 5.7502418, -14.0957165, 14.0957165
5: -7.1336541, 5.6869812, -7.1336541, 5.6869812, -12.8206348, 12.8206348
6: -6.3902140, 6.4361529, -6.3902140, 6.4361529, -12.8263664, 12.8263664
7: -7.1255641, 6.9407606, -7.1255641, 6.9407606, -14.0663242, 14.0663242
8: -10.3838263, 4.4998660, -10.3838263, 4.4998660, -14.8836918, 14.8836918
9: -6.2737536, 6.3332019, -6.2737536, 6.3332019, -12.6069536, 12.6069536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=60, inp2_unstable=60, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=230, inp2_unstable=230, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205413, upper bound: 12.3205414
time: 4.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205409, upper bound: 12.3205415
time: 4.05 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.0235176, 5.7524662, -7.0235176, 5.7524662, -12.7759838, 12.7759819
1: -6.1864815, 5.1664095, -6.1864815, 5.1664095, -11.3528910, 11.3528910
2: -7.9090343, 5.1656170, -7.9090343, 5.1656170, -13.0746517, 13.0746517
3: -8.4604549, 4.2214842, -8.4604549, 4.2214842, -12.6819391, 12.6819391
4: -8.3454752, 5.7502418, -8.3454752, 5.7502418, -14.0957165, 14.0957165
5: -7.1336541, 5.6869812, -7.1336541, 5.6869812, -12.8206348, 12.8206348
6: -6.3902140, 6.4361529, -6.3902140, 6.4361529, -12.8263664, 12.8263664
7: -7.1255641, 6.9407606, -7.1255641, 6.9407606, -14.0663242, 14.0663242
8: -10.3838263, 4.4998660, -10.3838263, 4.4998660, -14.8836918, 14.8836918
9: -6.2737536, 6.3332019, -6.2737536, 6.3332019, -12.6069536, 12.6069536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=60, inp2_unstable=60, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=230, inp2_unstable=230, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205413, upper bound: 12.3205414
time: 3.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205409, upper bound: 12.3205415
time: 4.06 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 9.44 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 9.44
Output dim: 8, lower bound: -12.3205418, upper bound: 12.3205409
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 9.44
Output dim: 8, lower bound: -12.3205409, upper bound: 12.3205414
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 9.44
Output dim: 8, lower bound: -12.3205418, upper bound: 12.3205410
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 9.44
Output dim: 8, lower bound: -12.3205409, upper bound: 12.3205414
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 9.44
Output dim: 8, lower bound: -12.3205413, upper bound: 12.3205414
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 9.44
Output dim: 8, lower bound: -12.3205409, upper bound: 12.3205415
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 9.44
Output dim: 8, lower bound: -12.3205413, upper bound: 12.3205414
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 9.44
Output dim: 8, lower bound: -12.3205409, upper bound: 12.3205415

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.0235176, 5.7524662, -7.0235176, 5.7524662, -12.7759838, 12.7759819
1: -6.1864815, 5.1664095, -6.1864815, 5.1664095, -11.3528910, 11.3528910
2: -7.9090343, 5.1656170, -7.9090343, 5.1656170, -13.0746517, 13.0746517
3: -8.4604549, 4.2214842, -8.4604549, 4.2214842, -12.6819391, 12.6819391
4: -8.3454752, 5.7502418, -8.3454752, 5.7502418, -14.0957165, 14.0957165
5: -7.1336541, 5.6869812, -7.1336541, 5.6869812, -12.8206348, 12.8206348
6: -6.3902140, 6.4361529, -6.3902140, 6.4361529, -12.8263664, 12.8263664
7: -7.1255641, 6.9407606, -7.1255641, 6.9407606, -14.0663242, 14.0663242
8: -10.3838263, 4.4998660, -10.3838263, 4.4998660, -14.8836918, 14.8836918
9: -6.2737536, 6.3332019, -6.2737536, 6.3332019, -12.6069536, 12.6069536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=60, inp2_unstable=60, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=230, inp2_unstable=230, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204811, upper bound: 12.3204803
time: 2.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204805, upper bound: 12.3204814
time: 2.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.0235176, 5.7524662, -7.0235176, 5.7524662, -12.7759838, 12.7759819
1: -6.1864815, 5.1664095, -6.1864815, 5.1664095, -11.3528910, 11.3528910
2: -7.9090343, 5.1656170, -7.9090343, 5.1656170, -13.0746517, 13.0746517
3: -8.4604549, 4.2214842, -8.4604549, 4.2214842, -12.6819391, 12.6819391
4: -8.3454752, 5.7502418, -8.3454752, 5.7502418, -14.0957165, 14.0957165
5: -7.1336541, 5.6869812, -7.1336541, 5.6869812, -12.8206348, 12.8206348
6: -6.3902140, 6.4361529, -6.3902140, 6.4361529, -12.8263664, 12.8263664
7: -7.1255641, 6.9407606, -7.1255641, 6.9407606, -14.0663242, 14.0663242
8: -10.3838263, 4.4998660, -10.3838263, 4.4998660, -14.8836918, 14.8836918
9: -6.2737536, 6.3332019, -6.2737536, 6.3332019, -12.6069536, 12.6069536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=60, inp2_unstable=60, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=230, inp2_unstable=230, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204809, upper bound: 12.3204804
time: 2.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204801, upper bound: 12.3204811
time: 2.44 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.0235176, 5.7524662, -7.0235176, 5.7524662, -12.7759838, 12.7759819
1: -6.1864815, 5.1664095, -6.1864815, 5.1664095, -11.3528910, 11.3528910
2: -7.9090343, 5.1656170, -7.9090343, 5.1656170, -13.0746517, 13.0746517
3: -8.4604549, 4.2214842, -8.4604549, 4.2214842, -12.6819391, 12.6819391
4: -8.3454752, 5.7502418, -8.3454752, 5.7502418, -14.0957165, 14.0957165
5: -7.1336541, 5.6869812, -7.1336541, 5.6869812, -12.8206348, 12.8206348
6: -6.3902140, 6.4361529, -6.3902140, 6.4361529, -12.8263664, 12.8263664
7: -7.1255641, 6.9407606, -7.1255641, 6.9407606, -14.0663242, 14.0663242
8: -10.3838263, 4.4998660, -10.3838263, 4.4998660, -14.8836918, 14.8836918
9: -6.2737536, 6.3332019, -6.2737536, 6.3332019, -12.6069536, 12.6069536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=60, inp2_unstable=60, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=230, inp2_unstable=230, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204811, upper bound: 12.3204803
time: 2.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204805, upper bound: 12.3204814
time: 2.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.0235176, 5.7524662, -7.0235176, 5.7524662, -12.7759838, 12.7759819
1: -6.1864815, 5.1664095, -6.1864815, 5.1664095, -11.3528910, 11.3528910
2: -7.9090343, 5.1656170, -7.9090343, 5.1656170, -13.0746517, 13.0746517
3: -8.4604549, 4.2214842, -8.4604549, 4.2214842, -12.6819391, 12.6819391
4: -8.3454752, 5.7502418, -8.3454752, 5.7502418, -14.0957165, 14.0957165
5: -7.1336541, 5.6869812, -7.1336541, 5.6869812, -12.8206348, 12.8206348
6: -6.3902140, 6.4361529, -6.3902140, 6.4361529, -12.8263664, 12.8263664
7: -7.1255641, 6.9407606, -7.1255641, 6.9407606, -14.0663242, 14.0663242
8: -10.3838263, 4.4998660, -10.3838263, 4.4998660, -14.8836918, 14.8836918
9: -6.2737536, 6.3332019, -6.2737536, 6.3332019, -12.6069536, 12.6069536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=60, inp2_unstable=60, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=230, inp2_unstable=230, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204809, upper bound: 12.3204804
time: 3.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204801, upper bound: 12.3204816
time: 2.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.0235176, 5.7524662, -7.0235176, 5.7524662, -12.7759838, 12.7759819
1: -6.1864815, 5.1664095, -6.1864815, 5.1664095, -11.3528910, 11.3528910
2: -7.9090343, 5.1656170, -7.9090343, 5.1656170, -13.0746517, 13.0746517
3: -8.4604549, 4.2214842, -8.4604549, 4.2214842, -12.6819391, 12.6819391
4: -8.3454752, 5.7502418, -8.3454752, 5.7502418, -14.0957165, 14.0957165
5: -7.1336541, 5.6869812, -7.1336541, 5.6869812, -12.8206348, 12.8206348
6: -6.3902140, 6.4361529, -6.3902140, 6.4361529, -12.8263664, 12.8263664
7: -7.1255641, 6.9407606, -7.1255641, 6.9407606, -14.0663242, 14.0663242
8: -10.3838263, 4.4998660, -10.3838263, 4.4998660, -14.8836918, 14.8836918
9: -6.2737536, 6.3332019, -6.2737536, 6.3332019, -12.6069536, 12.6069536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=60, inp2_unstable=60, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=230, inp2_unstable=230, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204811, upper bound: 12.3204803
time: 2.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204805, upper bound: 12.3204814
time: 3.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.0235176, 5.7524662, -7.0235176, 5.7524662, -12.7759838, 12.7759819
1: -6.1864815, 5.1664095, -6.1864815, 5.1664095, -11.3528910, 11.3528910
2: -7.9090343, 5.1656170, -7.9090343, 5.1656170, -13.0746517, 13.0746517
3: -8.4604549, 4.2214842, -8.4604549, 4.2214842, -12.6819391, 12.6819391
4: -8.3454752, 5.7502418, -8.3454752, 5.7502418, -14.0957165, 14.0957165
5: -7.1336541, 5.6869812, -7.1336541, 5.6869812, -12.8206348, 12.8206348
6: -6.3902140, 6.4361529, -6.3902140, 6.4361529, -12.8263664, 12.8263664
7: -7.1255641, 6.9407606, -7.1255641, 6.9407606, -14.0663242, 14.0663242
8: -10.3838263, 4.4998660, -10.3838263, 4.4998660, -14.8836918, 14.8836918
9: -6.2737536, 6.3332019, -6.2737536, 6.3332019, -12.6069536, 12.6069536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=60, inp2_unstable=60, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=230, inp2_unstable=230, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204814, upper bound: 12.3204804
time: 3.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204801, upper bound: 12.3204816
time: 3.01 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.0235176, 5.7524662, -7.0235176, 5.7524662, -12.7759838, 12.7759819
1: -6.1864815, 5.1664095, -6.1864815, 5.1664095, -11.3528910, 11.3528910
2: -7.9090343, 5.1656170, -7.9090343, 5.1656170, -13.0746517, 13.0746517
3: -8.4604549, 4.2214842, -8.4604549, 4.2214842, -12.6819391, 12.6819391
4: -8.3454752, 5.7502418, -8.3454752, 5.7502418, -14.0957165, 14.0957165
5: -7.1336541, 5.6869812, -7.1336541, 5.6869812, -12.8206348, 12.8206348
6: -6.3902140, 6.4361529, -6.3902140, 6.4361529, -12.8263664, 12.8263664
7: -7.1255641, 6.9407606, -7.1255641, 6.9407606, -14.0663242, 14.0663242
8: -10.3838263, 4.4998660, -10.3838263, 4.4998660, -14.8836918, 14.8836918
9: -6.2737536, 6.3332019, -6.2737536, 6.3332019, -12.6069536, 12.6069536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=60, inp2_unstable=60, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=230, inp2_unstable=230, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204811, upper bound: 12.3204803
time: 2.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204803, upper bound: 12.3204814
time: 3.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.0235176, 5.7524662, -7.0235176, 5.7524662, -12.7759838, 12.7759819
1: -6.1864815, 5.1664095, -6.1864815, 5.1664095, -11.3528910, 11.3528910
2: -7.9090343, 5.1656170, -7.9090343, 5.1656170, -13.0746517, 13.0746517
3: -8.4604549, 4.2214842, -8.4604549, 4.2214842, -12.6819391, 12.6819391
4: -8.3454752, 5.7502418, -8.3454752, 5.7502418, -14.0957165, 14.0957165
5: -7.1336541, 5.6869812, -7.1336541, 5.6869812, -12.8206348, 12.8206348
6: -6.3902140, 6.4361529, -6.3902140, 6.4361529, -12.8263664, 12.8263664
7: -7.1255641, 6.9407606, -7.1255641, 6.9407606, -14.0663242, 14.0663242
8: -10.3838263, 4.4998660, -10.3838263, 4.4998660, -14.8836918, 14.8836918
9: -6.2737536, 6.3332019, -6.2737536, 6.3332019, -12.6069536, 12.6069536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=60, inp2_unstable=60, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=230, inp2_unstable=230, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204814, upper bound: 12.3204804
time: 3.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204801, upper bound: 12.3204816
time: 3.02 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 7.42 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 7.42
Output dim: 8, lower bound: -12.3204811, upper bound: 12.3204803
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 7.42
Output dim: 8, lower bound: -12.3204805, upper bound: 12.3204814
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 7.42
Output dim: 8, lower bound: -12.3204809, upper bound: 12.3204804
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 7.42
Output dim: 8, lower bound: -12.3204801, upper bound: 12.3204811
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 7.42
Output dim: 8, lower bound: -12.3204811, upper bound: 12.3204803
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 7.42
Output dim: 8, lower bound: -12.3204805, upper bound: 12.3204814
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 7.42
Output dim: 8, lower bound: -12.3204809, upper bound: 12.3204804
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 7.42
Output dim: 8, lower bound: -12.3204801, upper bound: 12.3204816
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 7.42
Output dim: 8, lower bound: -12.3204811, upper bound: 12.3204803
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 7.42
Output dim: 8, lower bound: -12.3204805, upper bound: 12.3204814
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 7.42
Output dim: 8, lower bound: -12.3204814, upper bound: 12.3204804
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 7.42
Output dim: 8, lower bound: -12.3204801, upper bound: 12.3204816
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 7.42
Output dim: 8, lower bound: -12.3204811, upper bound: 12.3204803
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 7.42
Output dim: 8, lower bound: -12.3204803, upper bound: 12.3204814
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 7.42
Output dim: 8, lower bound: -12.3204814, upper bound: 12.3204804
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 7.42
Output dim: 8, lower bound: -12.3204801, upper bound: 12.3204816

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.0235176, 5.7524662, -7.0235176, 5.7524662, -12.7759838, 12.7759819
1: -6.1864815, 5.1664095, -6.1864815, 5.1664095, -11.3528910, 11.3528910
2: -7.9090343, 5.1656170, -7.9090343, 5.1656170, -13.0746517, 13.0746517
3: -8.4604549, 4.2214842, -8.4604549, 4.2214842, -12.6819391, 12.6819391
4: -8.3454752, 5.7502418, -8.3454752, 5.7502418, -14.0957165, 14.0957165
5: -7.1336541, 5.6869812, -7.1336541, 5.6869812, -12.8206348, 12.8206348
6: -6.3902140, 6.4361529, -6.3902140, 6.4361529, -12.8263664, 12.8263664
7: -7.1255641, 6.9407606, -7.1255641, 6.9407606, -14.0663242, 14.0663242
8: -10.3838263, 4.4998660, -10.3838263, 4.4998660, -14.8836918, 14.8836918
9: -6.2737536, 6.3332019, -6.2737536, 6.3332019, -12.6069536, 12.6069536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=60, inp2_unstable=60, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=230, inp2_unstable=230, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204816, upper bound: 12.3204800
time: 2.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204816, upper bound: 12.3204803
time: 2.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.0235176, 5.7524662, -7.0235176, 5.7524662, -12.7759838, 12.7759819
1: -6.1864815, 5.1664095, -6.1864815, 5.1664095, -11.3528910, 11.3528910
2: -7.9090343, 5.1656170, -7.9090343, 5.1656170, -13.0746517, 13.0746517
3: -8.4604549, 4.2214842, -8.4604549, 4.2214842, -12.6819391, 12.6819391
4: -8.3454752, 5.7502418, -8.3454752, 5.7502418, -14.0957165, 14.0957165
5: -7.1336541, 5.6869812, -7.1336541, 5.6869812, -12.8206348, 12.8206348
6: -6.3902140, 6.4361529, -6.3902140, 6.4361529, -12.8263664, 12.8263664
7: -7.1255641, 6.9407606, -7.1255641, 6.9407606, -14.0663242, 14.0663242
8: -10.3838263, 4.4998660, -10.3838263, 4.4998660, -14.8836918, 14.8836918
9: -6.2737536, 6.3332019, -6.2737536, 6.3332019, -12.6069536, 12.6069536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=60, inp2_unstable=60, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=230, inp2_unstable=230, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204803, upper bound: 12.3204811
time: 2.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204799, upper bound: 12.3204814
time: 2.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.0235176, 5.7524662, -7.0235176, 5.7524662, -12.7759838, 12.7759819
1: -6.1864815, 5.1664095, -6.1864815, 5.1664095, -11.3528910, 11.3528910
2: -7.9090343, 5.1656170, -7.9090343, 5.1656170, -13.0746517, 13.0746517
3: -8.4604549, 4.2214842, -8.4604549, 4.2214842, -12.6819391, 12.6819391
4: -8.3454752, 5.7502418, -8.3454752, 5.7502418, -14.0957165, 14.0957165
5: -7.1336541, 5.6869812, -7.1336541, 5.6869812, -12.8206348, 12.8206348
6: -6.3902140, 6.4361529, -6.3902140, 6.4361529, -12.8263664, 12.8263664
7: -7.1255641, 6.9407606, -7.1255641, 6.9407606, -14.0663242, 14.0663242
8: -10.3838263, 4.4998660, -10.3838263, 4.4998660, -14.8836918, 14.8836918
9: -6.2737536, 6.3332019, -6.2737536, 6.3332019, -12.6069536, 12.6069536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=60, inp2_unstable=60, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=230, inp2_unstable=230, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204809, upper bound: 12.3204804
time: 2.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204814, upper bound: 12.3204804
time: 3.91 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.0235176, 5.7524662, -7.0235176, 5.7524662, -12.7759838, 12.7759819
1: -6.1864815, 5.1664095, -6.1864815, 5.1664095, -11.3528910, 11.3528910
2: -7.9090343, 5.1656170, -7.9090343, 5.1656170, -13.0746517, 13.0746517
3: -8.4604549, 4.2214842, -8.4604549, 4.2214842, -12.6819391, 12.6819391
4: -8.3454752, 5.7502418, -8.3454752, 5.7502418, -14.0957165, 14.0957165
5: -7.1336541, 5.6869812, -7.1336541, 5.6869812, -12.8206348, 12.8206348
6: -6.3902140, 6.4361529, -6.3902140, 6.4361529, -12.8263664, 12.8263664
7: -7.1255641, 6.9407606, -7.1255641, 6.9407606, -14.0663242, 14.0663242
8: -10.3838263, 4.4998660, -10.3838263, 4.4998660, -14.8836918, 14.8836918
9: -6.2737536, 6.3332019, -6.2737536, 6.3332019, -12.6069536, 12.6069536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=60, inp2_unstable=60, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=230, inp2_unstable=230, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204803, upper bound: 12.3204816
time: 4.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204803, upper bound: 12.3204816
time: 2.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.0235176, 5.7524662, -7.0235176, 5.7524662, -12.7759838, 12.7759819
1: -6.1864815, 5.1664095, -6.1864815, 5.1664095, -11.3528910, 11.3528910
2: -7.9090343, 5.1656170, -7.9090343, 5.1656170, -13.0746517, 13.0746517
3: -8.4604549, 4.2214842, -8.4604549, 4.2214842, -12.6819391, 12.6819391
4: -8.3454752, 5.7502418, -8.3454752, 5.7502418, -14.0957165, 14.0957165
5: -7.1336541, 5.6869812, -7.1336541, 5.6869812, -12.8206348, 12.8206348
6: -6.3902140, 6.4361529, -6.3902140, 6.4361529, -12.8263664, 12.8263664
7: -7.1255641, 6.9407606, -7.1255641, 6.9407606, -14.0663242, 14.0663242
8: -10.3838263, 4.4998660, -10.3838263, 4.4998660, -14.8836918, 14.8836918
9: -6.2737536, 6.3332019, -6.2737536, 6.3332019, -12.6069536, 12.6069536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=60, inp2_unstable=60, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=230, inp2_unstable=230, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204816, upper bound: 12.3204800
time: 2.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204816, upper bound: 12.3204803
time: 2.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.0235176, 5.7524662, -7.0235176, 5.7524662, -12.7759838, 12.7759819
1: -6.1864815, 5.1664095, -6.1864815, 5.1664095, -11.3528910, 11.3528910
2: -7.9090343, 5.1656170, -7.9090343, 5.1656170, -13.0746517, 13.0746517
3: -8.4604549, 4.2214842, -8.4604549, 4.2214842, -12.6819391, 12.6819391
4: -8.3454752, 5.7502418, -8.3454752, 5.7502418, -14.0957165, 14.0957165
5: -7.1336541, 5.6869812, -7.1336541, 5.6869812, -12.8206348, 12.8206348
6: -6.3902140, 6.4361529, -6.3902140, 6.4361529, -12.8263664, 12.8263664
7: -7.1255641, 6.9407606, -7.1255641, 6.9407606, -14.0663242, 14.0663242
8: -10.3838263, 4.4998660, -10.3838263, 4.4998660, -14.8836918, 14.8836918
9: -6.2737536, 6.3332019, -6.2737536, 6.3332019, -12.6069536, 12.6069536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=60, inp2_unstable=60, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=230, inp2_unstable=230, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204804, upper bound: 12.3204809
time: 3.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204799, upper bound: 12.3204814
time: 2.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.0235176, 5.7524662, -7.0235176, 5.7524662, -12.7759838, 12.7759819
1: -6.1864815, 5.1664095, -6.1864815, 5.1664095, -11.3528910, 11.3528910
2: -7.9090343, 5.1656170, -7.9090343, 5.1656170, -13.0746517, 13.0746517
3: -8.4604549, 4.2214842, -8.4604549, 4.2214842, -12.6819391, 12.6819391
4: -8.3454752, 5.7502418, -8.3454752, 5.7502418, -14.0957165, 14.0957165
5: -7.1336541, 5.6869812, -7.1336541, 5.6869812, -12.8206348, 12.8206348
6: -6.3902140, 6.4361529, -6.3902140, 6.4361529, -12.8263664, 12.8263664
7: -7.1255641, 6.9407606, -7.1255641, 6.9407606, -14.0663242, 14.0663242
8: -10.3838263, 4.4998660, -10.3838263, 4.4998660, -14.8836918, 14.8836918
9: -6.2737536, 6.3332019, -6.2737536, 6.3332019, -12.6069536, 12.6069536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=60, inp2_unstable=60, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=230, inp2_unstable=230, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204809, upper bound: 12.3204804
time: 2.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204803, upper bound: 12.3204805
time: 3.07 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.0235176, 5.7524662, -7.0235176, 5.7524662, -12.7759838, 12.7759819
1: -6.1864815, 5.1664095, -6.1864815, 5.1664095, -11.3528910, 11.3528910
2: -7.9090343, 5.1656170, -7.9090343, 5.1656170, -13.0746517, 13.0746517
3: -8.4604549, 4.2214842, -8.4604549, 4.2214842, -12.6819391, 12.6819391
4: -8.3454752, 5.7502418, -8.3454752, 5.7502418, -14.0957165, 14.0957165
5: -7.1336541, 5.6869812, -7.1336541, 5.6869812, -12.8206348, 12.8206348
6: -6.3902140, 6.4361529, -6.3902140, 6.4361529, -12.8263664, 12.8263664
7: -7.1255641, 6.9407606, -7.1255641, 6.9407606, -14.0663242, 14.0663242
8: -10.3838263, 4.4998660, -10.3838263, 4.4998660, -14.8836918, 14.8836918
9: -6.2737536, 6.3332019, -6.2737536, 6.3332019, -12.6069536, 12.6069536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=60, inp2_unstable=60, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=230, inp2_unstable=230, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204803, upper bound: 12.3204816
time: 3.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204803, upper bound: 12.3204813
time: 2.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.0235176, 5.7524662, -7.0235176, 5.7524662, -12.7759838, 12.7759819
1: -6.1864815, 5.1664095, -6.1864815, 5.1664095, -11.3528910, 11.3528910
2: -7.9090343, 5.1656170, -7.9090343, 5.1656170, -13.0746517, 13.0746517
3: -8.4604549, 4.2214842, -8.4604549, 4.2214842, -12.6819391, 12.6819391
4: -8.3454752, 5.7502418, -8.3454752, 5.7502418, -14.0957165, 14.0957165
5: -7.1336541, 5.6869812, -7.1336541, 5.6869812, -12.8206348, 12.8206348
6: -6.3902140, 6.4361529, -6.3902140, 6.4361529, -12.8263664, 12.8263664
7: -7.1255641, 6.9407606, -7.1255641, 6.9407606, -14.0663242, 14.0663242
8: -10.3838263, 4.4998660, -10.3838263, 4.4998660, -14.8836918, 14.8836918
9: -6.2737536, 6.3332019, -6.2737536, 6.3332019, -12.6069536, 12.6069536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=60, inp2_unstable=60, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=230, inp2_unstable=230, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204816, upper bound: 12.3204802
time: 3.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204816, upper bound: 12.3204803
time: 2.28 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.0235176, 5.7524662, -7.0235176, 5.7524662, -12.7759838, 12.7759819
1: -6.1864815, 5.1664095, -6.1864815, 5.1664095, -11.3528910, 11.3528910
2: -7.9090343, 5.1656170, -7.9090343, 5.1656170, -13.0746517, 13.0746517
3: -8.4604549, 4.2214842, -8.4604549, 4.2214842, -12.6819391, 12.6819391
4: -8.3454752, 5.7502418, -8.3454752, 5.7502418, -14.0957165, 14.0957165
5: -7.1336541, 5.6869812, -7.1336541, 5.6869812, -12.8206348, 12.8206348
6: -6.3902140, 6.4361529, -6.3902140, 6.4361529, -12.8263664, 12.8263664
7: -7.1255641, 6.9407606, -7.1255641, 6.9407606, -14.0663242, 14.0663242
8: -10.3838263, 4.4998660, -10.3838263, 4.4998660, -14.8836918, 14.8836918
9: -6.2737536, 6.3332019, -6.2737536, 6.3332019, -12.6069536, 12.6069536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=60, inp2_unstable=60, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=230, inp2_unstable=230, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204805, upper bound: 12.3204814
time: 4.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204799, upper bound: 12.3204814
time: 2.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.0235176, 5.7524662, -7.0235176, 5.7524662, -12.7759838, 12.7759819
1: -6.1864815, 5.1664095, -6.1864815, 5.1664095, -11.3528910, 11.3528910
2: -7.9090343, 5.1656170, -7.9090343, 5.1656170, -13.0746517, 13.0746517
3: -8.4604549, 4.2214842, -8.4604549, 4.2214842, -12.6819391, 12.6819391
4: -8.3454752, 5.7502418, -8.3454752, 5.7502418, -14.0957165, 14.0957165
5: -7.1336541, 5.6869812, -7.1336541, 5.6869812, -12.8206348, 12.8206348
6: -6.3902140, 6.4361529, -6.3902140, 6.4361529, -12.8263664, 12.8263664
7: -7.1255641, 6.9407606, -7.1255641, 6.9407606, -14.0663242, 14.0663242
8: -10.3838263, 4.4998660, -10.3838263, 4.4998660, -14.8836918, 14.8836918
9: -6.2737536, 6.3332019, -6.2737536, 6.3332019, -12.6069536, 12.6069536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=60, inp2_unstable=60, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=230, inp2_unstable=230, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204809, upper bound: 12.3204804
time: 3.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204814, upper bound: 12.3204804
time: 2.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.0235176, 5.7524662, -7.0235176, 5.7524662, -12.7759838, 12.7759819
1: -6.1864815, 5.1664095, -6.1864815, 5.1664095, -11.3528910, 11.3528910
2: -7.9090343, 5.1656170, -7.9090343, 5.1656170, -13.0746517, 13.0746517
3: -8.4604549, 4.2214842, -8.4604549, 4.2214842, -12.6819391, 12.6819391
4: -8.3454752, 5.7502418, -8.3454752, 5.7502418, -14.0957165, 14.0957165
5: -7.1336541, 5.6869812, -7.1336541, 5.6869812, -12.8206348, 12.8206348
6: -6.3902140, 6.4361529, -6.3902140, 6.4361529, -12.8263664, 12.8263664
7: -7.1255641, 6.9407606, -7.1255641, 6.9407606, -14.0663242, 14.0663242
8: -10.3838263, 4.4998660, -10.3838263, 4.4998660, -14.8836918, 14.8836918
9: -6.2737536, 6.3332019, -6.2737536, 6.3332019, -12.6069536, 12.6069536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=60, inp2_unstable=60, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=230, inp2_unstable=230, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204803, upper bound: 12.3204816
time: 5.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204803, upper bound: 12.3204816
time: 2.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.0235176, 5.7524662, -7.0235176, 5.7524662, -12.7759838, 12.7759819
1: -6.1864815, 5.1664095, -6.1864815, 5.1664095, -11.3528910, 11.3528910
2: -7.9090343, 5.1656170, -7.9090343, 5.1656170, -13.0746517, 13.0746517
3: -8.4604549, 4.2214842, -8.4604549, 4.2214842, -12.6819391, 12.6819391
4: -8.3454752, 5.7502418, -8.3454752, 5.7502418, -14.0957165, 14.0957165
5: -7.1336541, 5.6869812, -7.1336541, 5.6869812, -12.8206348, 12.8206348
6: -6.3902140, 6.4361529, -6.3902140, 6.4361529, -12.8263664, 12.8263664
7: -7.1255641, 6.9407606, -7.1255641, 6.9407606, -14.0663242, 14.0663242
8: -10.3838263, 4.4998660, -10.3838263, 4.4998660, -14.8836918, 14.8836918
9: -6.2737536, 6.3332019, -6.2737536, 6.3332019, -12.6069536, 12.6069536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=60, inp2_unstable=60, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=230, inp2_unstable=230, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204816, upper bound: 12.3204802
time: 3.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204816, upper bound: 12.3204803
time: 2.28 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.0235176, 5.7524662, -7.0235176, 5.7524662, -12.7759838, 12.7759819
1: -6.1864815, 5.1664095, -6.1864815, 5.1664095, -11.3528910, 11.3528910
2: -7.9090343, 5.1656170, -7.9090343, 5.1656170, -13.0746517, 13.0746517
3: -8.4604549, 4.2214842, -8.4604549, 4.2214842, -12.6819391, 12.6819391
4: -8.3454752, 5.7502418, -8.3454752, 5.7502418, -14.0957165, 14.0957165
5: -7.1336541, 5.6869812, -7.1336541, 5.6869812, -12.8206348, 12.8206348
6: -6.3902140, 6.4361529, -6.3902140, 6.4361529, -12.8263664, 12.8263664
7: -7.1255641, 6.9407606, -7.1255641, 6.9407606, -14.0663242, 14.0663242
8: -10.3838263, 4.4998660, -10.3838263, 4.4998660, -14.8836918, 14.8836918
9: -6.2737536, 6.3332019, -6.2737536, 6.3332019, -12.6069536, 12.6069536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=60, inp2_unstable=60, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=230, inp2_unstable=230, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204800, upper bound: 12.3204814
time: 3.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204799, upper bound: 12.3204814
time: 2.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.0235176, 5.7524662, -7.0235176, 5.7524662, -12.7759838, 12.7759819
1: -6.1864815, 5.1664095, -6.1864815, 5.1664095, -11.3528910, 11.3528910
2: -7.9090343, 5.1656170, -7.9090343, 5.1656170, -13.0746517, 13.0746517
3: -8.4604549, 4.2214842, -8.4604549, 4.2214842, -12.6819391, 12.6819391
4: -8.3454752, 5.7502418, -8.3454752, 5.7502418, -14.0957165, 14.0957165
5: -7.1336541, 5.6869812, -7.1336541, 5.6869812, -12.8206348, 12.8206348
6: -6.3902140, 6.4361529, -6.3902140, 6.4361529, -12.8263664, 12.8263664
7: -7.1255641, 6.9407606, -7.1255641, 6.9407606, -14.0663242, 14.0663242
8: -10.3838263, 4.4998660, -10.3838263, 4.4998660, -14.8836918, 14.8836918
9: -6.2737536, 6.3332019, -6.2737536, 6.3332019, -12.6069536, 12.6069536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=60, inp2_unstable=60, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=230, inp2_unstable=230, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204809, upper bound: 12.3204804
time: 3.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204814, upper bound: 12.3204804
time: 2.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.0235176, 5.7524662, -7.0235176, 5.7524662, -12.7759838, 12.7759819
1: -6.1864815, 5.1664095, -6.1864815, 5.1664095, -11.3528910, 11.3528910
2: -7.9090343, 5.1656170, -7.9090343, 5.1656170, -13.0746517, 13.0746517
3: -8.4604549, 4.2214842, -8.4604549, 4.2214842, -12.6819391, 12.6819391
4: -8.3454752, 5.7502418, -8.3454752, 5.7502418, -14.0957165, 14.0957165
5: -7.1336541, 5.6869812, -7.1336541, 5.6869812, -12.8206348, 12.8206348
6: -6.3902140, 6.4361529, -6.3902140, 6.4361529, -12.8263664, 12.8263664
7: -7.1255641, 6.9407606, -7.1255641, 6.9407606, -14.0663242, 14.0663242
8: -10.3838263, 4.4998660, -10.3838263, 4.4998660, -14.8836918, 14.8836918
9: -6.2737536, 6.3332019, -6.2737536, 6.3332019, -12.6069536, 12.6069536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=60, inp2_unstable=60, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=230, inp2_unstable=230, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204803, upper bound: 12.3204816
time: 3.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204803, upper bound: 12.3204811
time: 2.31 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 9.49 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.49
Output dim: 8, lower bound: -12.3204816, upper bound: 12.3204800
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.49
Output dim: 8, lower bound: -12.3204816, upper bound: 12.3204803
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.49
Output dim: 8, lower bound: -12.3204803, upper bound: 12.3204811
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.49
Output dim: 8, lower bound: -12.3204799, upper bound: 12.3204814
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.49
Output dim: 8, lower bound: -12.3204809, upper bound: 12.3204804
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.49
Output dim: 8, lower bound: -12.3204814, upper bound: 12.3204804
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.49
Output dim: 8, lower bound: -12.3204803, upper bound: 12.3204816
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.49
Output dim: 8, lower bound: -12.3204803, upper bound: 12.3204816
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.49
Output dim: 8, lower bound: -12.3204816, upper bound: 12.3204800
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.49
Output dim: 8, lower bound: -12.3204816, upper bound: 12.3204803
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.49
Output dim: 8, lower bound: -12.3204804, upper bound: 12.3204809
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.49
Output dim: 8, lower bound: -12.3204799, upper bound: 12.3204814
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.49
Output dim: 8, lower bound: -12.3204809, upper bound: 12.3204804
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.49
Output dim: 8, lower bound: -12.3204803, upper bound: 12.3204805
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.49
Output dim: 8, lower bound: -12.3204803, upper bound: 12.3204816
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.49
Output dim: 8, lower bound: -12.3204803, upper bound: 12.3204813
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.49
Output dim: 8, lower bound: -12.3204816, upper bound: 12.3204802
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.49
Output dim: 8, lower bound: -12.3204816, upper bound: 12.3204803
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.49
Output dim: 8, lower bound: -12.3204805, upper bound: 12.3204814
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.49
Output dim: 8, lower bound: -12.3204799, upper bound: 12.3204814
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.49
Output dim: 8, lower bound: -12.3204809, upper bound: 12.3204804
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.49
Output dim: 8, lower bound: -12.3204814, upper bound: 12.3204804
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.49
Output dim: 8, lower bound: -12.3204803, upper bound: 12.3204816
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.49
Output dim: 8, lower bound: -12.3204803, upper bound: 12.3204816
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.49
Output dim: 8, lower bound: -12.3204816, upper bound: 12.3204802
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.49
Output dim: 8, lower bound: -12.3204816, upper bound: 12.3204803
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.49
Output dim: 8, lower bound: -12.3204800, upper bound: 12.3204814
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.49
Output dim: 8, lower bound: -12.3204799, upper bound: 12.3204814
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.49
Output dim: 8, lower bound: -12.3204809, upper bound: 12.3204804
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.49
Output dim: 8, lower bound: -12.3204814, upper bound: 12.3204804
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.49
Output dim: 8, lower bound: -12.3204803, upper bound: 12.3204816
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.49
Output dim: 8, lower bound: -12.3204803, upper bound: 12.3204811

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.0235176, 5.7524662, -7.0235176, 5.7524662, -12.7759838, 12.7759819
1: -6.1864815, 5.1664095, -6.1864815, 5.1664095, -11.3528910, 11.3528910
2: -7.9090343, 5.1656170, -7.9090343, 5.1656170, -13.0746517, 13.0746517
3: -8.4604549, 4.2214842, -8.4604549, 4.2214842, -12.6819391, 12.6819391
4: -8.3454752, 5.7502418, -8.3454752, 5.7502418, -14.0957165, 14.0957165
5: -7.1336541, 5.6869812, -7.1336541, 5.6869812, -12.8206348, 12.8206348
6: -6.3902140, 6.4361529, -6.3902140, 6.4361529, -12.8263664, 12.8263664
7: -7.1255641, 6.9407606, -7.1255641, 6.9407606, -14.0663242, 14.0663242
8: -10.3838263, 4.4998660, -10.3838263, 4.4998660, -14.8836918, 14.8836918
9: -6.2737536, 6.3332019, -6.2737536, 6.3332019, -12.6069536, 12.6069536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=60, inp2_unstable=60, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=230, inp2_unstable=230, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203836, upper bound: 12.3203824
time: 4.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203833, upper bound: 12.3203828
time: 3.15 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.0235176, 5.7524662, -7.0235176, 5.7524662, -12.7759838, 12.7759819
1: -6.1864815, 5.1664095, -6.1864815, 5.1664095, -11.3528910, 11.3528910
2: -7.9090343, 5.1656170, -7.9090343, 5.1656170, -13.0746517, 13.0746517
3: -8.4604549, 4.2214842, -8.4604549, 4.2214842, -12.6819391, 12.6819391
4: -8.3454752, 5.7502418, -8.3454752, 5.7502418, -14.0957165, 14.0957165
5: -7.1336541, 5.6869812, -7.1336541, 5.6869812, -12.8206348, 12.8206348
6: -6.3902140, 6.4361529, -6.3902140, 6.4361529, -12.8263664, 12.8263664
7: -7.1255641, 6.9407606, -7.1255641, 6.9407606, -14.0663242, 14.0663242
8: -10.3838263, 4.4998660, -10.3838263, 4.4998660, -14.8836918, 14.8836918
9: -6.2737536, 6.3332019, -6.2737536, 6.3332019, -12.6069536, 12.6069536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=60, inp2_unstable=60, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=230, inp2_unstable=230, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203836, upper bound: 12.3203824
time: 2.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203833, upper bound: 12.3203828
time: 7.51 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.0235176, 5.7524662, -7.0235176, 5.7524662, -12.7759838, 12.7759819
1: -6.1864815, 5.1664095, -6.1864815, 5.1664095, -11.3528910, 11.3528910
2: -7.9090343, 5.1656170, -7.9090343, 5.1656170, -13.0746517, 13.0746517
3: -8.4604549, 4.2214842, -8.4604549, 4.2214842, -12.6819391, 12.6819391
4: -8.3454752, 5.7502418, -8.3454752, 5.7502418, -14.0957165, 14.0957165
5: -7.1336541, 5.6869812, -7.1336541, 5.6869812, -12.8206348, 12.8206348
6: -6.3902140, 6.4361529, -6.3902140, 6.4361529, -12.8263664, 12.8263664
7: -7.1255641, 6.9407606, -7.1255641, 6.9407606, -14.0663242, 14.0663242
8: -10.3838263, 4.4998660, -10.3838263, 4.4998660, -14.8836918, 14.8836918
9: -6.2737536, 6.3332019, -6.2737536, 6.3332019, -12.6069536, 12.6069536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=60, inp2_unstable=60, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=230, inp2_unstable=230, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203828, upper bound: 12.3203833
time: 3.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203824, upper bound: 12.3203836
time: 3.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.0235176, 5.7524662, -7.0235176, 5.7524662, -12.7759838, 12.7759819
1: -6.1864815, 5.1664095, -6.1864815, 5.1664095, -11.3528910, 11.3528910
2: -7.9090343, 5.1656170, -7.9090343, 5.1656170, -13.0746517, 13.0746517
3: -8.4604549, 4.2214842, -8.4604549, 4.2214842, -12.6819391, 12.6819391
4: -8.3454752, 5.7502418, -8.3454752, 5.7502418, -14.0957165, 14.0957165
5: -7.1336541, 5.6869812, -7.1336541, 5.6869812, -12.8206348, 12.8206348
6: -6.3902140, 6.4361529, -6.3902140, 6.4361529, -12.8263664, 12.8263664
7: -7.1255641, 6.9407606, -7.1255641, 6.9407606, -14.0663242, 14.0663242
8: -10.3838263, 4.4998660, -10.3838263, 4.4998660, -14.8836918, 14.8836918
9: -6.2737536, 6.3332019, -6.2737536, 6.3332019, -12.6069536, 12.6069536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=60, inp2_unstable=60, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=230, inp2_unstable=230, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203828, upper bound: 12.3203833
time: 3.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203824, upper bound: 12.3203836
time: 3.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.0235176, 5.7524662, -7.0235176, 5.7524662, -12.7759838, 12.7759819
1: -6.1864815, 5.1664095, -6.1864815, 5.1664095, -11.3528910, 11.3528910
2: -7.9090343, 5.1656170, -7.9090343, 5.1656170, -13.0746517, 13.0746517
3: -8.4604549, 4.2214842, -8.4604549, 4.2214842, -12.6819391, 12.6819391
4: -8.3454752, 5.7502418, -8.3454752, 5.7502418, -14.0957165, 14.0957165
5: -7.1336541, 5.6869812, -7.1336541, 5.6869812, -12.8206348, 12.8206348
6: -6.3902140, 6.4361529, -6.3902140, 6.4361529, -12.8263664, 12.8263664
7: -7.1255641, 6.9407606, -7.1255641, 6.9407606, -14.0663242, 14.0663242
8: -10.3838263, 4.4998660, -10.3838263, 4.4998660, -14.8836918, 14.8836918
9: -6.2737536, 6.3332019, -6.2737536, 6.3332019, -12.6069536, 12.6069536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=60, inp2_unstable=60, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=230, inp2_unstable=230, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203836, upper bound: 12.3203824
time: 3.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203833, upper bound: 12.3203828
time: 2.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.0235176, 5.7524662, -7.0235176, 5.7524662, -12.7759838, 12.7759819
1: -6.1864815, 5.1664095, -6.1864815, 5.1664095, -11.3528910, 11.3528910
2: -7.9090343, 5.1656170, -7.9090343, 5.1656170, -13.0746517, 13.0746517
3: -8.4604549, 4.2214842, -8.4604549, 4.2214842, -12.6819391, 12.6819391
4: -8.3454752, 5.7502418, -8.3454752, 5.7502418, -14.0957165, 14.0957165
5: -7.1336541, 5.6869812, -7.1336541, 5.6869812, -12.8206348, 12.8206348
6: -6.3902140, 6.4361529, -6.3902140, 6.4361529, -12.8263664, 12.8263664
7: -7.1255641, 6.9407606, -7.1255641, 6.9407606, -14.0663242, 14.0663242
8: -10.3838263, 4.4998660, -10.3838263, 4.4998660, -14.8836918, 14.8836918
9: -6.2737536, 6.3332019, -6.2737536, 6.3332019, -12.6069536, 12.6069536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=60, inp2_unstable=60, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=230, inp2_unstable=230, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203836, upper bound: 12.3203824
time: 2.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203833, upper bound: 12.3203828
time: 4.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.0235176, 5.7524662, -7.0235176, 5.7524662, -12.7759838, 12.7759819
1: -6.1864815, 5.1664095, -6.1864815, 5.1664095, -11.3528910, 11.3528910
2: -7.9090343, 5.1656170, -7.9090343, 5.1656170, -13.0746517, 13.0746517
3: -8.4604549, 4.2214842, -8.4604549, 4.2214842, -12.6819391, 12.6819391
4: -8.3454752, 5.7502418, -8.3454752, 5.7502418, -14.0957165, 14.0957165
5: -7.1336541, 5.6869812, -7.1336541, 5.6869812, -12.8206348, 12.8206348
6: -6.3902140, 6.4361529, -6.3902140, 6.4361529, -12.8263664, 12.8263664
7: -7.1255641, 6.9407606, -7.1255641, 6.9407606, -14.0663242, 14.0663242
8: -10.3838263, 4.4998660, -10.3838263, 4.4998660, -14.8836918, 14.8836918
9: -6.2737536, 6.3332019, -6.2737536, 6.3332019, -12.6069536, 12.6069536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=60, inp2_unstable=60, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=230, inp2_unstable=230, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203828, upper bound: 12.3203833
time: 2.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203824, upper bound: 12.3203836
time: 4.18 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.0235176, 5.7524662, -7.0235176, 5.7524662, -12.7759838, 12.7759819
1: -6.1864815, 5.1664095, -6.1864815, 5.1664095, -11.3528910, 11.3528910
2: -7.9090343, 5.1656170, -7.9090343, 5.1656170, -13.0746517, 13.0746517
3: -8.4604549, 4.2214842, -8.4604549, 4.2214842, -12.6819391, 12.6819391
4: -8.3454752, 5.7502418, -8.3454752, 5.7502418, -14.0957165, 14.0957165
5: -7.1336541, 5.6869812, -7.1336541, 5.6869812, -12.8206348, 12.8206348
6: -6.3902140, 6.4361529, -6.3902140, 6.4361529, -12.8263664, 12.8263664
7: -7.1255641, 6.9407606, -7.1255641, 6.9407606, -14.0663242, 14.0663242
8: -10.3838263, 4.4998660, -10.3838263, 4.4998660, -14.8836918, 14.8836918
9: -6.2737536, 6.3332019, -6.2737536, 6.3332019, -12.6069536, 12.6069536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=60, inp2_unstable=60, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=230, inp2_unstable=230, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203828, upper bound: 12.3203833
time: 2.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203824, upper bound: 12.3203836
time: 4.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.0235176, 5.7524662, -7.0235176, 5.7524662, -12.7759838, 12.7759819
1: -6.1864815, 5.1664095, -6.1864815, 5.1664095, -11.3528910, 11.3528910
2: -7.9090343, 5.1656170, -7.9090343, 5.1656170, -13.0746517, 13.0746517
3: -8.4604549, 4.2214842, -8.4604549, 4.2214842, -12.6819391, 12.6819391
4: -8.3454752, 5.7502418, -8.3454752, 5.7502418, -14.0957165, 14.0957165
5: -7.1336541, 5.6869812, -7.1336541, 5.6869812, -12.8206348, 12.8206348
6: -6.3902140, 6.4361529, -6.3902140, 6.4361529, -12.8263664, 12.8263664
7: -7.1255641, 6.9407606, -7.1255641, 6.9407606, -14.0663242, 14.0663242
8: -10.3838263, 4.4998660, -10.3838263, 4.4998660, -14.8836918, 14.8836918
9: -6.2737536, 6.3332019, -6.2737536, 6.3332019, -12.6069536, 12.6069536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=60, inp2_unstable=60, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=230, inp2_unstable=230, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203836, upper bound: 12.3203824
time: 3.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203833, upper bound: 12.3203829
time: 4.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.0235176, 5.7524662, -7.0235176, 5.7524662, -12.7759838, 12.7759819
1: -6.1864815, 5.1664095, -6.1864815, 5.1664095, -11.3528910, 11.3528910
2: -7.9090343, 5.1656170, -7.9090343, 5.1656170, -13.0746517, 13.0746517
3: -8.4604549, 4.2214842, -8.4604549, 4.2214842, -12.6819391, 12.6819391
4: -8.3454752, 5.7502418, -8.3454752, 5.7502418, -14.0957165, 14.0957165
5: -7.1336541, 5.6869812, -7.1336541, 5.6869812, -12.8206348, 12.8206348
6: -6.3902140, 6.4361529, -6.3902140, 6.4361529, -12.8263664, 12.8263664
7: -7.1255641, 6.9407606, -7.1255641, 6.9407606, -14.0663242, 14.0663242
8: -10.3838263, 4.4998660, -10.3838263, 4.4998660, -14.8836918, 14.8836918
9: -6.2737536, 6.3332019, -6.2737536, 6.3332019, -12.6069536, 12.6069536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=60, inp2_unstable=60, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=230, inp2_unstable=230, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203836, upper bound: 12.3203824
time: 5.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203833, upper bound: 12.3203828
time: 3.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.0235176, 5.7524662, -7.0235176, 5.7524662, -12.7759838, 12.7759819
1: -6.1864815, 5.1664095, -6.1864815, 5.1664095, -11.3528910, 11.3528910
2: -7.9090343, 5.1656170, -7.9090343, 5.1656170, -13.0746517, 13.0746517
3: -8.4604549, 4.2214842, -8.4604549, 4.2214842, -12.6819391, 12.6819391
4: -8.3454752, 5.7502418, -8.3454752, 5.7502418, -14.0957165, 14.0957165
5: -7.1336541, 5.6869812, -7.1336541, 5.6869812, -12.8206348, 12.8206348
6: -6.3902140, 6.4361529, -6.3902140, 6.4361529, -12.8263664, 12.8263664
7: -7.1255641, 6.9407606, -7.1255641, 6.9407606, -14.0663242, 14.0663242
8: -10.3838263, 4.4998660, -10.3838263, 4.4998660, -14.8836918, 14.8836918
9: -6.2737536, 6.3332019, -6.2737536, 6.3332019, -12.6069536, 12.6069536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=60, inp2_unstable=60, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=230, inp2_unstable=230, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203828, upper bound: 12.3203833
time: 4.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203824, upper bound: 12.3203836
time: 3.06 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.0235176, 5.7524662, -7.0235176, 5.7524662, -12.7759838, 12.7759819
1: -6.1864815, 5.1664095, -6.1864815, 5.1664095, -11.3528910, 11.3528910
2: -7.9090343, 5.1656170, -7.9090343, 5.1656170, -13.0746517, 13.0746517
3: -8.4604549, 4.2214842, -8.4604549, 4.2214842, -12.6819391, 12.6819391
4: -8.3454752, 5.7502418, -8.3454752, 5.7502418, -14.0957165, 14.0957165
5: -7.1336541, 5.6869812, -7.1336541, 5.6869812, -12.8206348, 12.8206348
6: -6.3902140, 6.4361529, -6.3902140, 6.4361529, -12.8263664, 12.8263664
7: -7.1255641, 6.9407606, -7.1255641, 6.9407606, -14.0663242, 14.0663242
8: -10.3838263, 4.4998660, -10.3838263, 4.4998660, -14.8836918, 14.8836918
9: -6.2737536, 6.3332019, -6.2737536, 6.3332019, -12.6069536, 12.6069536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=60, inp2_unstable=60, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=230, inp2_unstable=230, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203828, upper bound: 12.3203833
time: 3.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203824, upper bound: 12.3203836
time: 3.27 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.0235176, 5.7524662, -7.0235176, 5.7524662, -12.7759838, 12.7759819
1: -6.1864815, 5.1664095, -6.1864815, 5.1664095, -11.3528910, 11.3528910
2: -7.9090343, 5.1656170, -7.9090343, 5.1656170, -13.0746517, 13.0746517
3: -8.4604549, 4.2214842, -8.4604549, 4.2214842, -12.6819391, 12.6819391
4: -8.3454752, 5.7502418, -8.3454752, 5.7502418, -14.0957165, 14.0957165
5: -7.1336541, 5.6869812, -7.1336541, 5.6869812, -12.8206348, 12.8206348
6: -6.3902140, 6.4361529, -6.3902140, 6.4361529, -12.8263664, 12.8263664
7: -7.1255641, 6.9407606, -7.1255641, 6.9407606, -14.0663242, 14.0663242
8: -10.3838263, 4.4998660, -10.3838263, 4.4998660, -14.8836918, 14.8836918
9: -6.2737536, 6.3332019, -6.2737536, 6.3332019, -12.6069536, 12.6069536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=60, inp2_unstable=60, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=230, inp2_unstable=230, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203836, upper bound: 12.3203824
time: 3.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203833, upper bound: 12.3203828
time: 3.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.0235176, 5.7524662, -7.0235176, 5.7524662, -12.7759838, 12.7759819
1: -6.1864815, 5.1664095, -6.1864815, 5.1664095, -11.3528910, 11.3528910
2: -7.9090343, 5.1656170, -7.9090343, 5.1656170, -13.0746517, 13.0746517
3: -8.4604549, 4.2214842, -8.4604549, 4.2214842, -12.6819391, 12.6819391
4: -8.3454752, 5.7502418, -8.3454752, 5.7502418, -14.0957165, 14.0957165
5: -7.1336541, 5.6869812, -7.1336541, 5.6869812, -12.8206348, 12.8206348
6: -6.3902140, 6.4361529, -6.3902140, 6.4361529, -12.8263664, 12.8263664
7: -7.1255641, 6.9407606, -7.1255641, 6.9407606, -14.0663242, 14.0663242
8: -10.3838263, 4.4998660, -10.3838263, 4.4998660, -14.8836918, 14.8836918
9: -6.2737536, 6.3332019, -6.2737536, 6.3332019, -12.6069536, 12.6069536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=60, inp2_unstable=60, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=230, inp2_unstable=230, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203836, upper bound: 12.3203824
time: 2.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203824, upper bound: 12.3203828
time: 3.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.0235176, 5.7524662, -7.0235176, 5.7524662, -12.7759838, 12.7759819
1: -6.1864815, 5.1664095, -6.1864815, 5.1664095, -11.3528910, 11.3528910
2: -7.9090343, 5.1656170, -7.9090343, 5.1656170, -13.0746517, 13.0746517
3: -8.4604549, 4.2214842, -8.4604549, 4.2214842, -12.6819391, 12.6819391
4: -8.3454752, 5.7502418, -8.3454752, 5.7502418, -14.0957165, 14.0957165
5: -7.1336541, 5.6869812, -7.1336541, 5.6869812, -12.8206348, 12.8206348
6: -6.3902140, 6.4361529, -6.3902140, 6.4361529, -12.8263664, 12.8263664
7: -7.1255641, 6.9407606, -7.1255641, 6.9407606, -14.0663242, 14.0663242
8: -10.3838263, 4.4998660, -10.3838263, 4.4998660, -14.8836918, 14.8836918
9: -6.2737536, 6.3332019, -6.2737536, 6.3332019, -12.6069536, 12.6069536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=60, inp2_unstable=60, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=230, inp2_unstable=230, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203828, upper bound: 12.3203833
time: 2.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203824, upper bound: 12.3203836
time: 3.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.0235176, 5.7524662, -7.0235176, 5.7524662, -12.7759838, 12.7759819
1: -6.1864815, 5.1664095, -6.1864815, 5.1664095, -11.3528910, 11.3528910
2: -7.9090343, 5.1656170, -7.9090343, 5.1656170, -13.0746517, 13.0746517
3: -8.4604549, 4.2214842, -8.4604549, 4.2214842, -12.6819391, 12.6819391
4: -8.3454752, 5.7502418, -8.3454752, 5.7502418, -14.0957165, 14.0957165
5: -7.1336541, 5.6869812, -7.1336541, 5.6869812, -12.8206348, 12.8206348
6: -6.3902140, 6.4361529, -6.3902140, 6.4361529, -12.8263664, 12.8263664
7: -7.1255641, 6.9407606, -7.1255641, 6.9407606, -14.0663242, 14.0663242
8: -10.3838263, 4.4998660, -10.3838263, 4.4998660, -14.8836918, 14.8836918
9: -6.2737536, 6.3332019, -6.2737536, 6.3332019, -12.6069536, 12.6069536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=60, inp2_unstable=60, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=230, inp2_unstable=230, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203828, upper bound: 12.3203833
time: 2.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203824, upper bound: 12.3203836
time: 5.27 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.0235176, 5.7524662, -7.0235176, 5.7524662, -12.7759838, 12.7759819
1: -6.1864815, 5.1664095, -6.1864815, 5.1664095, -11.3528910, 11.3528910
2: -7.9090343, 5.1656170, -7.9090343, 5.1656170, -13.0746517, 13.0746517
3: -8.4604549, 4.2214842, -8.4604549, 4.2214842, -12.6819391, 12.6819391
4: -8.3454752, 5.7502418, -8.3454752, 5.7502418, -14.0957165, 14.0957165
5: -7.1336541, 5.6869812, -7.1336541, 5.6869812, -12.8206348, 12.8206348
6: -6.3902140, 6.4361529, -6.3902140, 6.4361529, -12.8263664, 12.8263664
7: -7.1255641, 6.9407606, -7.1255641, 6.9407606, -14.0663242, 14.0663242
8: -10.3838263, 4.4998660, -10.3838263, 4.4998660, -14.8836918, 14.8836918
9: -6.2737536, 6.3332019, -6.2737536, 6.3332019, -12.6069536, 12.6069536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=60, inp2_unstable=60, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=230, inp2_unstable=230, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203836, upper bound: 12.3203824
time: 3.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203833, upper bound: 12.3203828
time: 3.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.0235176, 5.7524662, -7.0235176, 5.7524662, -12.7759838, 12.7759819
1: -6.1864815, 5.1664095, -6.1864815, 5.1664095, -11.3528910, 11.3528910
2: -7.9090343, 5.1656170, -7.9090343, 5.1656170, -13.0746517, 13.0746517
3: -8.4604549, 4.2214842, -8.4604549, 4.2214842, -12.6819391, 12.6819391
4: -8.3454752, 5.7502418, -8.3454752, 5.7502418, -14.0957165, 14.0957165
5: -7.1336541, 5.6869812, -7.1336541, 5.6869812, -12.8206348, 12.8206348
6: -6.3902140, 6.4361529, -6.3902140, 6.4361529, -12.8263664, 12.8263664
7: -7.1255641, 6.9407606, -7.1255641, 6.9407606, -14.0663242, 14.0663242
8: -10.3838263, 4.4998660, -10.3838263, 4.4998660, -14.8836918, 14.8836918
9: -6.2737536, 6.3332019, -6.2737536, 6.3332019, -12.6069536, 12.6069536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=60, inp2_unstable=60, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=230, inp2_unstable=230, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203836, upper bound: 12.3203824
time: 2.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203833, upper bound: 12.3203828
time: 4.21 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.0235176, 5.7524662, -7.0235176, 5.7524662, -12.7759838, 12.7759819
1: -6.1864815, 5.1664095, -6.1864815, 5.1664095, -11.3528910, 11.3528910
2: -7.9090343, 5.1656170, -7.9090343, 5.1656170, -13.0746517, 13.0746517
3: -8.4604549, 4.2214842, -8.4604549, 4.2214842, -12.6819391, 12.6819391
4: -8.3454752, 5.7502418, -8.3454752, 5.7502418, -14.0957165, 14.0957165
5: -7.1336541, 5.6869812, -7.1336541, 5.6869812, -12.8206348, 12.8206348
6: -6.3902140, 6.4361529, -6.3902140, 6.4361529, -12.8263664, 12.8263664
7: -7.1255641, 6.9407606, -7.1255641, 6.9407606, -14.0663242, 14.0663242
8: -10.3838263, 4.4998660, -10.3838263, 4.4998660, -14.8836918, 14.8836918
9: -6.2737536, 6.3332019, -6.2737536, 6.3332019, -12.6069536, 12.6069536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=60, inp2_unstable=60, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=230, inp2_unstable=230, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203828, upper bound: 12.3203833
time: 3.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203824, upper bound: 12.3203836
time: 3.48 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.0235176, 5.7524662, -7.0235176, 5.7524662, -12.7759838, 12.7759819
1: -6.1864815, 5.1664095, -6.1864815, 5.1664095, -11.3528910, 11.3528910
2: -7.9090343, 5.1656170, -7.9090343, 5.1656170, -13.0746517, 13.0746517
3: -8.4604549, 4.2214842, -8.4604549, 4.2214842, -12.6819391, 12.6819391
4: -8.3454752, 5.7502418, -8.3454752, 5.7502418, -14.0957165, 14.0957165
5: -7.1336541, 5.6869812, -7.1336541, 5.6869812, -12.8206348, 12.8206348
6: -6.3902140, 6.4361529, -6.3902140, 6.4361529, -12.8263664, 12.8263664
7: -7.1255641, 6.9407606, -7.1255641, 6.9407606, -14.0663242, 14.0663242
8: -10.3838263, 4.4998660, -10.3838263, 4.4998660, -14.8836918, 14.8836918
9: -6.2737536, 6.3332019, -6.2737536, 6.3332019, -12.6069536, 12.6069536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=60, inp2_unstable=60, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=230, inp2_unstable=230, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203828, upper bound: 12.3203833
time: 2.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203824, upper bound: 12.3203836
time: 3.03 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.0235176, 5.7524662, -7.0235176, 5.7524662, -12.7759838, 12.7759819
1: -6.1864815, 5.1664095, -6.1864815, 5.1664095, -11.3528910, 11.3528910
2: -7.9090343, 5.1656170, -7.9090343, 5.1656170, -13.0746517, 13.0746517
3: -8.4604549, 4.2214842, -8.4604549, 4.2214842, -12.6819391, 12.6819391
4: -8.3454752, 5.7502418, -8.3454752, 5.7502418, -14.0957165, 14.0957165
5: -7.1336541, 5.6869812, -7.1336541, 5.6869812, -12.8206348, 12.8206348
6: -6.3902140, 6.4361529, -6.3902140, 6.4361529, -12.8263664, 12.8263664
7: -7.1255641, 6.9407606, -7.1255641, 6.9407606, -14.0663242, 14.0663242
8: -10.3838263, 4.4998660, -10.3838263, 4.4998660, -14.8836918, 14.8836918
9: -6.2737536, 6.3332019, -6.2737536, 6.3332019, -12.6069536, 12.6069536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=60, inp2_unstable=60, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=230, inp2_unstable=230, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203836, upper bound: 12.3203824
time: 3.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203833, upper bound: 12.3203828
time: 2.87 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.0235176, 5.7524662, -7.0235176, 5.7524662, -12.7759838, 12.7759819
1: -6.1864815, 5.1664095, -6.1864815, 5.1664095, -11.3528910, 11.3528910
2: -7.9090343, 5.1656170, -7.9090343, 5.1656170, -13.0746517, 13.0746517
3: -8.4604549, 4.2214842, -8.4604549, 4.2214842, -12.6819391, 12.6819391
4: -8.3454752, 5.7502418, -8.3454752, 5.7502418, -14.0957165, 14.0957165
5: -7.1336541, 5.6869812, -7.1336541, 5.6869812, -12.8206348, 12.8206348
6: -6.3902140, 6.4361529, -6.3902140, 6.4361529, -12.8263664, 12.8263664
7: -7.1255641, 6.9407606, -7.1255641, 6.9407606, -14.0663242, 14.0663242
8: -10.3838263, 4.4998660, -10.3838263, 4.4998660, -14.8836918, 14.8836918
9: -6.2737536, 6.3332019, -6.2737536, 6.3332019, -12.6069536, 12.6069536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=60, inp2_unstable=60, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=230, inp2_unstable=230, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203836, upper bound: 12.3203824
time: 3.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203833, upper bound: 12.3203828
time: 4.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.0235176, 5.7524662, -7.0235176, 5.7524662, -12.7759838, 12.7759819
1: -6.1864815, 5.1664095, -6.1864815, 5.1664095, -11.3528910, 11.3528910
2: -7.9090343, 5.1656170, -7.9090343, 5.1656170, -13.0746517, 13.0746517
3: -8.4604549, 4.2214842, -8.4604549, 4.2214842, -12.6819391, 12.6819391
4: -8.3454752, 5.7502418, -8.3454752, 5.7502418, -14.0957165, 14.0957165
5: -7.1336541, 5.6869812, -7.1336541, 5.6869812, -12.8206348, 12.8206348
6: -6.3902140, 6.4361529, -6.3902140, 6.4361529, -12.8263664, 12.8263664
7: -7.1255641, 6.9407606, -7.1255641, 6.9407606, -14.0663242, 14.0663242
8: -10.3838263, 4.4998660, -10.3838263, 4.4998660, -14.8836918, 14.8836918
9: -6.2737536, 6.3332019, -6.2737536, 6.3332019, -12.6069536, 12.6069536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=60, inp2_unstable=60, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=230, inp2_unstable=230, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203828, upper bound: 12.3203833
time: 4.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203824, upper bound: 12.3203836
time: 2.89 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.0235176, 5.7524662, -7.0235176, 5.7524662, -12.7759838, 12.7759819
1: -6.1864815, 5.1664095, -6.1864815, 5.1664095, -11.3528910, 11.3528910
2: -7.9090343, 5.1656170, -7.9090343, 5.1656170, -13.0746517, 13.0746517
3: -8.4604549, 4.2214842, -8.4604549, 4.2214842, -12.6819391, 12.6819391
4: -8.3454752, 5.7502418, -8.3454752, 5.7502418, -14.0957165, 14.0957165
5: -7.1336541, 5.6869812, -7.1336541, 5.6869812, -12.8206348, 12.8206348
6: -6.3902140, 6.4361529, -6.3902140, 6.4361529, -12.8263664, 12.8263664
7: -7.1255641, 6.9407606, -7.1255641, 6.9407606, -14.0663242, 14.0663242
8: -10.3838263, 4.4998660, -10.3838263, 4.4998660, -14.8836918, 14.8836918
9: -6.2737536, 6.3332019, -6.2737536, 6.3332019, -12.6069536, 12.6069536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=60, inp2_unstable=60, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=230, inp2_unstable=230, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203829, upper bound: 12.3203833
time: 3.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203824, upper bound: 12.3203836
time: 3.11 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.0235176, 5.7524662, -7.0235176, 5.7524662, -12.7759838, 12.7759819
1: -6.1864815, 5.1664095, -6.1864815, 5.1664095, -11.3528910, 11.3528910
2: -7.9090343, 5.1656170, -7.9090343, 5.1656170, -13.0746517, 13.0746517
3: -8.4604549, 4.2214842, -8.4604549, 4.2214842, -12.6819391, 12.6819391
4: -8.3454752, 5.7502418, -8.3454752, 5.7502418, -14.0957165, 14.0957165
5: -7.1336541, 5.6869812, -7.1336541, 5.6869812, -12.8206348, 12.8206348
6: -6.3902140, 6.4361529, -6.3902140, 6.4361529, -12.8263664, 12.8263664
7: -7.1255641, 6.9407606, -7.1255641, 6.9407606, -14.0663242, 14.0663242
8: -10.3838263, 4.4998660, -10.3838263, 4.4998660, -14.8836918, 14.8836918
9: -6.2737536, 6.3332019, -6.2737536, 6.3332019, -12.6069536, 12.6069536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=60, inp2_unstable=60, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=230, inp2_unstable=230, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203836, upper bound: 12.3203824
time: 3.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203833, upper bound: 12.3203828
time: 3.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.0235176, 5.7524662, -7.0235176, 5.7524662, -12.7759838, 12.7759819
1: -6.1864815, 5.1664095, -6.1864815, 5.1664095, -11.3528910, 11.3528910
2: -7.9090343, 5.1656170, -7.9090343, 5.1656170, -13.0746517, 13.0746517
3: -8.4604549, 4.2214842, -8.4604549, 4.2214842, -12.6819391, 12.6819391
4: -8.3454752, 5.7502418, -8.3454752, 5.7502418, -14.0957165, 14.0957165
5: -7.1336541, 5.6869812, -7.1336541, 5.6869812, -12.8206348, 12.8206348
6: -6.3902140, 6.4361529, -6.3902140, 6.4361529, -12.8263664, 12.8263664
7: -7.1255641, 6.9407606, -7.1255641, 6.9407606, -14.0663242, 14.0663242
8: -10.3838263, 4.4998660, -10.3838263, 4.4998660, -14.8836918, 14.8836918
9: -6.2737536, 6.3332019, -6.2737536, 6.3332019, -12.6069536, 12.6069536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=60, inp2_unstable=60, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=230, inp2_unstable=230, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203836, upper bound: 12.3203824
time: 3.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203833, upper bound: 12.3203828
time: 4.55 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.0235176, 5.7524662, -7.0235176, 5.7524662, -12.7759838, 12.7759819
1: -6.1864815, 5.1664095, -6.1864815, 5.1664095, -11.3528910, 11.3528910
2: -7.9090343, 5.1656170, -7.9090343, 5.1656170, -13.0746517, 13.0746517
3: -8.4604549, 4.2214842, -8.4604549, 4.2214842, -12.6819391, 12.6819391
4: -8.3454752, 5.7502418, -8.3454752, 5.7502418, -14.0957165, 14.0957165
5: -7.1336541, 5.6869812, -7.1336541, 5.6869812, -12.8206348, 12.8206348
6: -6.3902140, 6.4361529, -6.3902140, 6.4361529, -12.8263664, 12.8263664
7: -7.1255641, 6.9407606, -7.1255641, 6.9407606, -14.0663242, 14.0663242
8: -10.3838263, 4.4998660, -10.3838263, 4.4998660, -14.8836918, 14.8836918
9: -6.2737536, 6.3332019, -6.2737536, 6.3332019, -12.6069536, 12.6069536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=60, inp2_unstable=60, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=230, inp2_unstable=230, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203828, upper bound: 12.3203833
time: 3.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203824, upper bound: 12.3203836
time: 3.08 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.0235176, 5.7524662, -7.0235176, 5.7524662, -12.7759838, 12.7759819
1: -6.1864815, 5.1664095, -6.1864815, 5.1664095, -11.3528910, 11.3528910
2: -7.9090343, 5.1656170, -7.9090343, 5.1656170, -13.0746517, 13.0746517
3: -8.4604549, 4.2214842, -8.4604549, 4.2214842, -12.6819391, 12.6819391
4: -8.3454752, 5.7502418, -8.3454752, 5.7502418, -14.0957165, 14.0957165
5: -7.1336541, 5.6869812, -7.1336541, 5.6869812, -12.8206348, 12.8206348
6: -6.3902140, 6.4361529, -6.3902140, 6.4361529, -12.8263664, 12.8263664
7: -7.1255641, 6.9407606, -7.1255641, 6.9407606, -14.0663242, 14.0663242
8: -10.3838263, 4.4998660, -10.3838263, 4.4998660, -14.8836918, 14.8836918
9: -6.2737536, 6.3332019, -6.2737536, 6.3332019, -12.6069536, 12.6069536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=60, inp2_unstable=60, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=230, inp2_unstable=230, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203828, upper bound: 12.3203833
time: 2.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203824, upper bound: 12.3203836
time: 3.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.0235176, 5.7524662, -7.0235176, 5.7524662, -12.7759838, 12.7759819
1: -6.1864815, 5.1664095, -6.1864815, 5.1664095, -11.3528910, 11.3528910
2: -7.9090343, 5.1656170, -7.9090343, 5.1656170, -13.0746517, 13.0746517
3: -8.4604549, 4.2214842, -8.4604549, 4.2214842, -12.6819391, 12.6819391
4: -8.3454752, 5.7502418, -8.3454752, 5.7502418, -14.0957165, 14.0957165
5: -7.1336541, 5.6869812, -7.1336541, 5.6869812, -12.8206348, 12.8206348
6: -6.3902140, 6.4361529, -6.3902140, 6.4361529, -12.8263664, 12.8263664
7: -7.1255641, 6.9407606, -7.1255641, 6.9407606, -14.0663242, 14.0663242
8: -10.3838263, 4.4998660, -10.3838263, 4.4998660, -14.8836918, 14.8836918
9: -6.2737536, 6.3332019, -6.2737536, 6.3332019, -12.6069536, 12.6069536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=60, inp2_unstable=60, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=230, inp2_unstable=230, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203836, upper bound: 12.3203824
time: 3.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203833, upper bound: 12.3203828
time: 3.14 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.0235176, 5.7524662, -7.0235176, 5.7524662, -12.7759838, 12.7759819
1: -6.1864815, 5.1664095, -6.1864815, 5.1664095, -11.3528910, 11.3528910
2: -7.9090343, 5.1656170, -7.9090343, 5.1656170, -13.0746517, 13.0746517
3: -8.4604549, 4.2214842, -8.4604549, 4.2214842, -12.6819391, 12.6819391
4: -8.3454752, 5.7502418, -8.3454752, 5.7502418, -14.0957165, 14.0957165
5: -7.1336541, 5.6869812, -7.1336541, 5.6869812, -12.8206348, 12.8206348
6: -6.3902140, 6.4361529, -6.3902140, 6.4361529, -12.8263664, 12.8263664
7: -7.1255641, 6.9407606, -7.1255641, 6.9407606, -14.0663242, 14.0663242
8: -10.3838263, 4.4998660, -10.3838263, 4.4998660, -14.8836918, 14.8836918
9: -6.2737536, 6.3332019, -6.2737536, 6.3332019, -12.6069536, 12.6069536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=60, inp2_unstable=60, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=230, inp2_unstable=230, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203836, upper bound: 12.3203824
time: 3.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203833, upper bound: 12.3203828
time: 3.52 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.0235176, 5.7524662, -7.0235176, 5.7524662, -12.7759838, 12.7759819
1: -6.1864815, 5.1664095, -6.1864815, 5.1664095, -11.3528910, 11.3528910
2: -7.9090343, 5.1656170, -7.9090343, 5.1656170, -13.0746517, 13.0746517
3: -8.4604549, 4.2214842, -8.4604549, 4.2214842, -12.6819391, 12.6819391
4: -8.3454752, 5.7502418, -8.3454752, 5.7502418, -14.0957165, 14.0957165
5: -7.1336541, 5.6869812, -7.1336541, 5.6869812, -12.8206348, 12.8206348
6: -6.3902140, 6.4361529, -6.3902140, 6.4361529, -12.8263664, 12.8263664
7: -7.1255641, 6.9407606, -7.1255641, 6.9407606, -14.0663242, 14.0663242
8: -10.3838263, 4.4998660, -10.3838263, 4.4998660, -14.8836918, 14.8836918
9: -6.2737536, 6.3332019, -6.2737536, 6.3332019, -12.6069536, 12.6069536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=60, inp2_unstable=60, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=230, inp2_unstable=230, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203828, upper bound: 12.3203833
time: 3.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203824, upper bound: 12.3203836
time: 2.76 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 9.70 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.70
Output dim: 8, lower bound: -12.3203836, upper bound: 12.3203824
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.70
Output dim: 8, lower bound: -12.3203833, upper bound: 12.3203828
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.70
Output dim: 8, lower bound: -12.3203836, upper bound: 12.3203824
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.70
Output dim: 8, lower bound: -12.3203833, upper bound: 12.3203828
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.70
Output dim: 8, lower bound: -12.3203828, upper bound: 12.3203833
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.70
Output dim: 8, lower bound: -12.3203824, upper bound: 12.3203836
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.70
Output dim: 8, lower bound: -12.3203828, upper bound: 12.3203833
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.70
Output dim: 8, lower bound: -12.3203824, upper bound: 12.3203836
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.70
Output dim: 8, lower bound: -12.3203836, upper bound: 12.3203824
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.70
Output dim: 8, lower bound: -12.3203833, upper bound: 12.3203828
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.70
Output dim: 8, lower bound: -12.3203836, upper bound: 12.3203824
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.70
Output dim: 8, lower bound: -12.3203833, upper bound: 12.3203828
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.70
Output dim: 8, lower bound: -12.3203828, upper bound: 12.3203833
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.70
Output dim: 8, lower bound: -12.3203824, upper bound: 12.3203836
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.70
Output dim: 8, lower bound: -12.3203828, upper bound: 12.3203833
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.70
Output dim: 8, lower bound: -12.3203824, upper bound: 12.3203836
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.70
Output dim: 8, lower bound: -12.3203836, upper bound: 12.3203824
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.70
Output dim: 8, lower bound: -12.3203833, upper bound: 12.3203829
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.70
Output dim: 8, lower bound: -12.3203836, upper bound: 12.3203824
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.70
Output dim: 8, lower bound: -12.3203833, upper bound: 12.3203828
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.70
Output dim: 8, lower bound: -12.3203828, upper bound: 12.3203833
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.70
Output dim: 8, lower bound: -12.3203824, upper bound: 12.3203836
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.70
Output dim: 8, lower bound: -12.3203828, upper bound: 12.3203833
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.70
Output dim: 8, lower bound: -12.3203824, upper bound: 12.3203836
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.70
Output dim: 8, lower bound: -12.3203836, upper bound: 12.3203824
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.70
Output dim: 8, lower bound: -12.3203833, upper bound: 12.3203828
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.70
Output dim: 8, lower bound: -12.3203836, upper bound: 12.3203824
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.70
Output dim: 8, lower bound: -12.3203824, upper bound: 12.3203828
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.70
Output dim: 8, lower bound: -12.3203828, upper bound: 12.3203833
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.70
Output dim: 8, lower bound: -12.3203824, upper bound: 12.3203836
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.70
Output dim: 8, lower bound: -12.3203828, upper bound: 12.3203833
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.70
Output dim: 8, lower bound: -12.3203824, upper bound: 12.3203836
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.70
Output dim: 8, lower bound: -12.3203836, upper bound: 12.3203824
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.70
Output dim: 8, lower bound: -12.3203833, upper bound: 12.3203828
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.70
Output dim: 8, lower bound: -12.3203836, upper bound: 12.3203824
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.70
Output dim: 8, lower bound: -12.3203833, upper bound: 12.3203828
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.70
Output dim: 8, lower bound: -12.3203828, upper bound: 12.3203833
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.70
Output dim: 8, lower bound: -12.3203824, upper bound: 12.3203836
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.70
Output dim: 8, lower bound: -12.3203828, upper bound: 12.3203833
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.70
Output dim: 8, lower bound: -12.3203824, upper bound: 12.3203836
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.70
Output dim: 8, lower bound: -12.3203836, upper bound: 12.3203824
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.70
Output dim: 8, lower bound: -12.3203833, upper bound: 12.3203828
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.70
Output dim: 8, lower bound: -12.3203836, upper bound: 12.3203824
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.70
Output dim: 8, lower bound: -12.3203833, upper bound: 12.3203828
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.70
Output dim: 8, lower bound: -12.3203828, upper bound: 12.3203833
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.70
Output dim: 8, lower bound: -12.3203824, upper bound: 12.3203836
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.70
Output dim: 8, lower bound: -12.3203829, upper bound: 12.3203833
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.70
Output dim: 8, lower bound: -12.3203824, upper bound: 12.3203836
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.70
Output dim: 8, lower bound: -12.3203836, upper bound: 12.3203824
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.70
Output dim: 8, lower bound: -12.3203833, upper bound: 12.3203828
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.70
Output dim: 8, lower bound: -12.3203836, upper bound: 12.3203824
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.70
Output dim: 8, lower bound: -12.3203833, upper bound: 12.3203828
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.70
Output dim: 8, lower bound: -12.3203828, upper bound: 12.3203833
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.70
Output dim: 8, lower bound: -12.3203824, upper bound: 12.3203836
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.70
Output dim: 8, lower bound: -12.3203828, upper bound: 12.3203833
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.70
Output dim: 8, lower bound: -12.3203824, upper bound: 12.3203836
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.70
Output dim: 8, lower bound: -12.3203836, upper bound: 12.3203824
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.70
Output dim: 8, lower bound: -12.3203833, upper bound: 12.3203828
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.70
Output dim: 8, lower bound: -12.3203836, upper bound: 12.3203824
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.70
Output dim: 8, lower bound: -12.3203833, upper bound: 12.3203828
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.70
Output dim: 8, lower bound: -12.3203828, upper bound: 12.3203833
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.70
Output dim: 8, lower bound: -12.3203824, upper bound: 12.3203836
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.70
Output dim: 8, lower bound: -12.3204803, upper bound: 12.3204811

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 6.65 + 595.21 = 601.86 seconds
