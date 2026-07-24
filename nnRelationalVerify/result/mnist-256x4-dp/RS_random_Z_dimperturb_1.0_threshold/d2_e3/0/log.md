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
execution time: IAR + RelationalAnalysis = 1.30 + 5.26 = 6.56 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -12.3209782, upper bound: 12.3209782

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3209781, upper bound: 12.3209775
time: 4.49 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3209782, upper bound: 12.3209781
time: 2.89 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 7.40 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 7.40
Output dim: 8, lower bound: -12.3209781, upper bound: 12.3209775
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 7.40
Output dim: 8, lower bound: -12.3209782, upper bound: 12.3209781

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3201039, upper bound: 12.3201041
time: 2.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3201039, upper bound: 12.3201040
time: 2.44 seconds

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
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3209730, upper bound: 12.3209730
time: 2.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3209730, upper bound: 12.3209730
time: 3.00 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 6.88 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 6.88
Output dim: 8, lower bound: -12.3201039, upper bound: 12.3201041
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 6.88
Output dim: 8, lower bound: -12.3201039, upper bound: 12.3201040
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 6.88
Output dim: 8, lower bound: -12.3209730, upper bound: 12.3209730
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 6.88
Output dim: 8, lower bound: -12.3209730, upper bound: 12.3209730

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3194354, upper bound: 12.3194355
time: 3.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3194355, upper bound: 12.3194353
time: 1.66 seconds

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
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3200482, upper bound: 12.3200484
time: 2.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3200482, upper bound: 12.3200484
time: 2.10 seconds

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3209349, upper bound: 12.3209345
time: 3.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3209349, upper bound: 12.3209349
time: 3.47 seconds

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

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3209730, upper bound: 12.3209729
time: 4.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3209730, upper bound: 12.3209723
time: 4.01 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 9.57 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 9.57
Output dim: 8, lower bound: -12.3194354, upper bound: 12.3194355
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 9.57
Output dim: 8, lower bound: -12.3194355, upper bound: 12.3194353
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 9.57
Output dim: 8, lower bound: -12.3200482, upper bound: 12.3200484
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 9.57
Output dim: 8, lower bound: -12.3200482, upper bound: 12.3200484
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 9.57
Output dim: 8, lower bound: -12.3209349, upper bound: 12.3209345
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 9.57
Output dim: 8, lower bound: -12.3209349, upper bound: 12.3209349
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 9.57
Output dim: 8, lower bound: -12.3209730, upper bound: 12.3209729
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 9.57
Output dim: 8, lower bound: -12.3209730, upper bound: 12.3209723

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

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3187433, upper bound: 12.3187431
time: 2.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3187433, upper bound: 12.3187432
time: 2.29 seconds

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
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3194355, upper bound: 12.3194354
time: 1.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3194355, upper bound: 12.3194353
time: 3.15 seconds

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

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3200483, upper bound: 12.3200482
time: 2.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3200482, upper bound: 12.3200484
time: 2.59 seconds

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

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3200393, upper bound: 12.3200394
time: 3.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3200393, upper bound: 12.3200391
time: 2.02 seconds

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
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3209048, upper bound: 12.3209042
time: 3.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3209044, upper bound: 12.3209046
time: 4.97 seconds

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3209332, upper bound: 12.3209327
time: 3.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3209332, upper bound: 12.3209327
time: 3.39 seconds

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3209350, upper bound: 12.3209344
time: 4.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3209350, upper bound: 12.3209349
time: 3.67 seconds

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3209072, upper bound: 12.3209066
time: 3.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3209072, upper bound: 12.3209065
time: 3.53 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 8.59 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 8.59
Output dim: 8, lower bound: -12.3187433, upper bound: 12.3187431
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 8.59
Output dim: 8, lower bound: -12.3187433, upper bound: 12.3187432
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 8.59
Output dim: 8, lower bound: -12.3194355, upper bound: 12.3194354
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 8.59
Output dim: 8, lower bound: -12.3194355, upper bound: 12.3194353
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 8.59
Output dim: 8, lower bound: -12.3200483, upper bound: 12.3200482
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 8.59
Output dim: 8, lower bound: -12.3200482, upper bound: 12.3200484
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 8.59
Output dim: 8, lower bound: -12.3200393, upper bound: 12.3200394
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 8.59
Output dim: 8, lower bound: -12.3200393, upper bound: 12.3200391
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 8.59
Output dim: 8, lower bound: -12.3209048, upper bound: 12.3209042
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 8.59
Output dim: 8, lower bound: -12.3209044, upper bound: 12.3209046
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 8.59
Output dim: 8, lower bound: -12.3209332, upper bound: 12.3209327
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 8.59
Output dim: 8, lower bound: -12.3209332, upper bound: 12.3209327
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 8.59
Output dim: 8, lower bound: -12.3209350, upper bound: 12.3209344
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 8.59
Output dim: 8, lower bound: -12.3209350, upper bound: 12.3209349
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 8.59
Output dim: 8, lower bound: -12.3209072, upper bound: 12.3209066
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 8.59
Output dim: 8, lower bound: -12.3209072, upper bound: 12.3209065

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 175

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3187346, upper bound: 12.3187344
time: 2.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3187346, upper bound: 12.3187344
time: 2.40 seconds

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3185073, upper bound: 12.3185076
time: 2.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3185073, upper bound: 12.3185076
time: 2.10 seconds

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
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3187434, upper bound: 12.3187433
time: 3.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3187432, upper bound: 12.3187434
time: 2.82 seconds

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
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3194352, upper bound: 12.3194352
time: 2.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3194352, upper bound: 12.3194353
time: 2.13 seconds

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 169

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3200392, upper bound: 12.3200391
time: 1.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3200393, upper bound: 12.3200391
time: 1.80 seconds

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 88

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3192261, upper bound: 12.3192265
time: 2.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3192263, upper bound: 12.3192265
time: 3.50 seconds

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
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3192265, upper bound: 12.3192266
time: 2.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3192266, upper bound: 12.3192266
time: 2.21 seconds

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3192266, upper bound: 12.3192266
time: 2.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3192266, upper bound: 12.3192265
time: 3.20 seconds

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

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 169

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3207782, upper bound: 12.3207783
time: 3.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3207782, upper bound: 12.3207783
time: 4.60 seconds

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3208697, upper bound: 12.3208700
time: 3.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3208696, upper bound: 12.3208700
time: 4.66 seconds

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3209332, upper bound: 12.3209327
time: 3.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3209332, upper bound: 12.3209332
time: 3.30 seconds

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3208533, upper bound: 12.3208530
time: 4.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3208530, upper bound: 12.3208532
time: 4.67 seconds

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3209260, upper bound: 12.3209259
time: 3.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3209260, upper bound: 12.3209253
time: 3.53 seconds

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

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3208888, upper bound: 12.3208883
time: 4.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3208887, upper bound: 12.3208887
time: 3.63 seconds

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204996, upper bound: 12.3204994
time: 3.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204996, upper bound: 12.3204996
time: 2.98 seconds

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

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3208424, upper bound: 12.3208423
time: 3.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3208424, upper bound: 12.3208422
time: 5.25 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 10.25 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 10.25
Output dim: 8, lower bound: -12.3187346, upper bound: 12.3187344
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 10.25
Output dim: 8, lower bound: -12.3187346, upper bound: 12.3187344
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 10.25
Output dim: 8, lower bound: -12.3185073, upper bound: 12.3185076
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 10.25
Output dim: 8, lower bound: -12.3185073, upper bound: 12.3185076
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 10.25
Output dim: 8, lower bound: -12.3187434, upper bound: 12.3187433
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 10.25
Output dim: 8, lower bound: -12.3187432, upper bound: 12.3187434
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 10.25
Output dim: 8, lower bound: -12.3194352, upper bound: 12.3194352
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 10.25
Output dim: 8, lower bound: -12.3194352, upper bound: 12.3194353
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 10.25
Output dim: 8, lower bound: -12.3200392, upper bound: 12.3200391
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 10.25
Output dim: 8, lower bound: -12.3200393, upper bound: 12.3200391
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 10.25
Output dim: 8, lower bound: -12.3192261, upper bound: 12.3192265
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 10.25
Output dim: 8, lower bound: -12.3192263, upper bound: 12.3192265
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 10.25
Output dim: 8, lower bound: -12.3192265, upper bound: 12.3192266
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 10.25
Output dim: 8, lower bound: -12.3192266, upper bound: 12.3192266
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 10.25
Output dim: 8, lower bound: -12.3192266, upper bound: 12.3192266
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 10.25
Output dim: 8, lower bound: -12.3192266, upper bound: 12.3192265
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 10.25
Output dim: 8, lower bound: -12.3207782, upper bound: 12.3207783
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 10.25
Output dim: 8, lower bound: -12.3207782, upper bound: 12.3207783
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 10.25
Output dim: 8, lower bound: -12.3208697, upper bound: 12.3208700
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 10.25
Output dim: 8, lower bound: -12.3208696, upper bound: 12.3208700
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 10.25
Output dim: 8, lower bound: -12.3209332, upper bound: 12.3209327
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 10.25
Output dim: 8, lower bound: -12.3209332, upper bound: 12.3209332
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 10.25
Output dim: 8, lower bound: -12.3208533, upper bound: 12.3208530
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 10.25
Output dim: 8, lower bound: -12.3208530, upper bound: 12.3208532
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 10.25
Output dim: 8, lower bound: -12.3209260, upper bound: 12.3209259
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 10.25
Output dim: 8, lower bound: -12.3209260, upper bound: 12.3209253
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 10.25
Output dim: 8, lower bound: -12.3208888, upper bound: 12.3208883
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 10.25
Output dim: 8, lower bound: -12.3208887, upper bound: 12.3208887
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 10.25
Output dim: 8, lower bound: -12.3204996, upper bound: 12.3204994
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 10.25
Output dim: 8, lower bound: -12.3204996, upper bound: 12.3204996
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 10.25
Output dim: 8, lower bound: -12.3208424, upper bound: 12.3208423
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 10.25
Output dim: 8, lower bound: -12.3208424, upper bound: 12.3208422

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3184832, upper bound: 12.3184822
time: 2.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3184826, upper bound: 12.3184830
time: 2.07 seconds

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 188

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 107

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3178139, upper bound: 12.3178132
time: 2.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3178134, upper bound: 12.3178137
time: 1.83 seconds

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 188

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3174164, upper bound: 12.3174160
time: 2.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3174160, upper bound: 12.3174164
time: 2.09 seconds

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
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 175

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3174164, upper bound: 12.3174160
time: 1.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3174160, upper bound: 12.3174164
time: 1.82 seconds

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3187433, upper bound: 12.3187433
time: 2.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3187432, upper bound: 12.3187434
time: 2.47 seconds

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

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 175

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3187344, upper bound: 12.3187346
time: 2.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3187345, upper bound: 12.3187346
time: 3.16 seconds

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

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3188809, upper bound: 12.3188809
time: 3.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3188809, upper bound: 12.3188808
time: 2.06 seconds

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 107

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3190909, upper bound: 12.3190903
time: 4.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3190901, upper bound: 12.3190910
time: 2.64 seconds

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

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3200392, upper bound: 12.3200391
time: 1.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3200390, upper bound: 12.3200391
time: 1.72 seconds

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

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 169

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3200392, upper bound: 12.3200389
time: 2.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3200391, upper bound: 12.3200391
time: 4.35 seconds

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

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3189703, upper bound: 12.3189701
time: 1.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3189700, upper bound: 12.3189705
time: 3.43 seconds

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 169

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3192262, upper bound: 12.3192263
time: 1.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3192258, upper bound: 12.3192264
time: 1.76 seconds

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3186690, upper bound: 12.3186692
time: 1.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3186691, upper bound: 12.3186691
time: 2.91 seconds

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
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3184988, upper bound: 12.3184991
time: 2.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3184988, upper bound: 12.3184992
time: 2.88 seconds

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

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3186692, upper bound: 12.3186691
time: 1.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3186692, upper bound: 12.3186690
time: 3.05 seconds

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
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 175

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 188

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3189807, upper bound: 12.3189804
time: 2.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3189805, upper bound: 12.3189807
time: 1.84 seconds

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
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 188

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3206109, upper bound: 12.3206107
time: 4.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3206109, upper bound: 12.3206107
time: 4.18 seconds

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

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3207406, upper bound: 12.3207391
time: 3.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3207393, upper bound: 12.3207401
time: 3.60 seconds

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

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3208697, upper bound: 12.3208699
time: 4.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3208697, upper bound: 12.3208701
time: 3.52 seconds

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
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3207328, upper bound: 12.3207324
time: 3.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3207328, upper bound: 12.3207331
time: 4.76 seconds

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

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3209243, upper bound: 12.3209244
time: 3.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3209246, upper bound: 12.3209243
time: 4.62 seconds

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205981, upper bound: 12.3205985
time: 3.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205981, upper bound: 12.3205985
time: 3.32 seconds

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

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3208029, upper bound: 12.3208030
time: 5.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3208029, upper bound: 12.3208030
time: 5.81 seconds

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

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3208012, upper bound: 12.3207990
time: 3.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3207990, upper bound: 12.3208006
time: 4.72 seconds

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

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3206156, upper bound: 12.3206160
time: 4.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3206156, upper bound: 12.3206160
time: 4.13 seconds

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
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3209052, upper bound: 12.3209052
time: 5.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3209052, upper bound: 12.3209047
time: 4.11 seconds

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
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3208510, upper bound: 12.3208503
time: 4.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3208501, upper bound: 12.3208502
time: 2.80 seconds

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

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3208395, upper bound: 12.3208364
time: 5.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3208366, upper bound: 12.3208391
time: 3.70 seconds

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
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 188

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204312, upper bound: 12.3204310
time: 3.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204311, upper bound: 12.3204312
time: 3.92 seconds

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
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204996, upper bound: 12.3204992
time: 4.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204994, upper bound: 12.3204996
time: 6.97 seconds

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

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204678, upper bound: 12.3204678
time: 2.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204678, upper bound: 12.3204678
time: 2.41 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 169

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3208424, upper bound: 12.3208423
time: 4.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3208424, upper bound: 12.3208423
time: 3.75 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 9.39 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.39
Output dim: 8, lower bound: -12.3184832, upper bound: 12.3184822
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.39
Output dim: 8, lower bound: -12.3184826, upper bound: 12.3184830
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.39
Output dim: 8, lower bound: -12.3178139, upper bound: 12.3178132
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.39
Output dim: 8, lower bound: -12.3178134, upper bound: 12.3178137
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.39
Output dim: 8, lower bound: -12.3174164, upper bound: 12.3174160
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.39
Output dim: 8, lower bound: -12.3174160, upper bound: 12.3174164
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.39
Output dim: 8, lower bound: -12.3174164, upper bound: 12.3174160
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.39
Output dim: 8, lower bound: -12.3174160, upper bound: 12.3174164
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.39
Output dim: 8, lower bound: -12.3187433, upper bound: 12.3187433
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.39
Output dim: 8, lower bound: -12.3187432, upper bound: 12.3187434
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.39
Output dim: 8, lower bound: -12.3187344, upper bound: 12.3187346
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.39
Output dim: 8, lower bound: -12.3187345, upper bound: 12.3187346
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.39
Output dim: 8, lower bound: -12.3188809, upper bound: 12.3188809
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.39
Output dim: 8, lower bound: -12.3188809, upper bound: 12.3188808
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.39
Output dim: 8, lower bound: -12.3190909, upper bound: 12.3190903
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.39
Output dim: 8, lower bound: -12.3190901, upper bound: 12.3190910
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.39
Output dim: 8, lower bound: -12.3200392, upper bound: 12.3200391
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.39
Output dim: 8, lower bound: -12.3200390, upper bound: 12.3200391
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.39
Output dim: 8, lower bound: -12.3200392, upper bound: 12.3200389
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.39
Output dim: 8, lower bound: -12.3200391, upper bound: 12.3200391
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.39
Output dim: 8, lower bound: -12.3189703, upper bound: 12.3189701
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.39
Output dim: 8, lower bound: -12.3189700, upper bound: 12.3189705
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.39
Output dim: 8, lower bound: -12.3192262, upper bound: 12.3192263
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.39
Output dim: 8, lower bound: -12.3192258, upper bound: 12.3192264
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.39
Output dim: 8, lower bound: -12.3186690, upper bound: 12.3186692
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.39
Output dim: 8, lower bound: -12.3186691, upper bound: 12.3186691
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.39
Output dim: 8, lower bound: -12.3184988, upper bound: 12.3184991
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.39
Output dim: 8, lower bound: -12.3184988, upper bound: 12.3184992
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.39
Output dim: 8, lower bound: -12.3186692, upper bound: 12.3186691
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.39
Output dim: 8, lower bound: -12.3186692, upper bound: 12.3186690
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.39
Output dim: 8, lower bound: -12.3189807, upper bound: 12.3189804
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.39
Output dim: 8, lower bound: -12.3189805, upper bound: 12.3189807
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.39
Output dim: 8, lower bound: -12.3206109, upper bound: 12.3206107
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.39
Output dim: 8, lower bound: -12.3206109, upper bound: 12.3206107
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.39
Output dim: 8, lower bound: -12.3207406, upper bound: 12.3207391
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.39
Output dim: 8, lower bound: -12.3207393, upper bound: 12.3207401
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.39
Output dim: 8, lower bound: -12.3208697, upper bound: 12.3208699
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.39
Output dim: 8, lower bound: -12.3208697, upper bound: 12.3208701
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.39
Output dim: 8, lower bound: -12.3207328, upper bound: 12.3207324
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.39
Output dim: 8, lower bound: -12.3207328, upper bound: 12.3207331
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.39
Output dim: 8, lower bound: -12.3209243, upper bound: 12.3209244
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.39
Output dim: 8, lower bound: -12.3209246, upper bound: 12.3209243
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.39
Output dim: 8, lower bound: -12.3205981, upper bound: 12.3205985
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.39
Output dim: 8, lower bound: -12.3205981, upper bound: 12.3205985
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.39
Output dim: 8, lower bound: -12.3208029, upper bound: 12.3208030
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.39
Output dim: 8, lower bound: -12.3208029, upper bound: 12.3208030
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.39
Output dim: 8, lower bound: -12.3208012, upper bound: 12.3207990
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.39
Output dim: 8, lower bound: -12.3207990, upper bound: 12.3208006
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.39
Output dim: 8, lower bound: -12.3206156, upper bound: 12.3206160
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.39
Output dim: 8, lower bound: -12.3206156, upper bound: 12.3206160
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.39
Output dim: 8, lower bound: -12.3209052, upper bound: 12.3209052
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.39
Output dim: 8, lower bound: -12.3209052, upper bound: 12.3209047
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.39
Output dim: 8, lower bound: -12.3208510, upper bound: 12.3208503
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.39
Output dim: 8, lower bound: -12.3208501, upper bound: 12.3208502
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.39
Output dim: 8, lower bound: -12.3208395, upper bound: 12.3208364
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.39
Output dim: 8, lower bound: -12.3208366, upper bound: 12.3208391
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.39
Output dim: 8, lower bound: -12.3204312, upper bound: 12.3204310
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.39
Output dim: 8, lower bound: -12.3204311, upper bound: 12.3204312
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.39
Output dim: 8, lower bound: -12.3204996, upper bound: 12.3204992
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.39
Output dim: 8, lower bound: -12.3204994, upper bound: 12.3204996
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.39
Output dim: 8, lower bound: -12.3204678, upper bound: 12.3204678
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.39
Output dim: 8, lower bound: -12.3204678, upper bound: 12.3204678
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.39
Output dim: 8, lower bound: -12.3208424, upper bound: 12.3208423
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.39
Output dim: 8, lower bound: -12.3208424, upper bound: 12.3208423

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=60, inp2_unstable=60, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=230, inp2_unstable=230, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3174062, upper bound: 12.3174060
time: 2.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3174061, upper bound: 12.3174060
time: 2.74 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 8.99 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 8.99
Output dim: 8, lower bound: -12.3174062, upper bound: 12.3174060
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 8.99
Output dim: 8, lower bound: -12.3174061, upper bound: 12.3174060
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 8, lower bound: -12.3184826, upper bound: 12.3184830
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 8, lower bound: -12.3178139, upper bound: 12.3178132
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 8, lower bound: -12.3178134, upper bound: 12.3178137
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 8, lower bound: -12.3174164, upper bound: 12.3174160
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 8, lower bound: -12.3174160, upper bound: 12.3174164
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 8, lower bound: -12.3174164, upper bound: 12.3174160
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 8, lower bound: -12.3174160, upper bound: 12.3174164
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 8, lower bound: -12.3187433, upper bound: 12.3187433
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 8, lower bound: -12.3187432, upper bound: 12.3187434
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 8, lower bound: -12.3187344, upper bound: 12.3187346
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 8, lower bound: -12.3187345, upper bound: 12.3187346
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 8, lower bound: -12.3188809, upper bound: 12.3188809
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 8, lower bound: -12.3188809, upper bound: 12.3188808
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 8, lower bound: -12.3190909, upper bound: 12.3190903
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 8, lower bound: -12.3190901, upper bound: 12.3190910
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 8, lower bound: -12.3200392, upper bound: 12.3200391
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 8, lower bound: -12.3200390, upper bound: 12.3200391
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 8, lower bound: -12.3200392, upper bound: 12.3200389
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 8, lower bound: -12.3200391, upper bound: 12.3200391
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 8, lower bound: -12.3189703, upper bound: 12.3189701
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 8, lower bound: -12.3189700, upper bound: 12.3189705
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 8, lower bound: -12.3192262, upper bound: 12.3192263
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 8, lower bound: -12.3192258, upper bound: 12.3192264
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 8, lower bound: -12.3186690, upper bound: 12.3186692
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 8, lower bound: -12.3186691, upper bound: 12.3186691
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 8, lower bound: -12.3184988, upper bound: 12.3184991
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 8, lower bound: -12.3184988, upper bound: 12.3184992
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 8, lower bound: -12.3186692, upper bound: 12.3186691
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 8, lower bound: -12.3186692, upper bound: 12.3186690
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 8, lower bound: -12.3189807, upper bound: 12.3189804
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 8, lower bound: -12.3189805, upper bound: 12.3189807
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 8, lower bound: -12.3206109, upper bound: 12.3206107
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 8, lower bound: -12.3206109, upper bound: 12.3206107
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 8, lower bound: -12.3207406, upper bound: 12.3207391
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 8, lower bound: -12.3207393, upper bound: 12.3207401
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 8, lower bound: -12.3208697, upper bound: 12.3208699
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 8, lower bound: -12.3208697, upper bound: 12.3208701
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 8, lower bound: -12.3207328, upper bound: 12.3207324
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 8, lower bound: -12.3207328, upper bound: 12.3207331
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 8, lower bound: -12.3209243, upper bound: 12.3209244
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 8, lower bound: -12.3209246, upper bound: 12.3209243
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 8, lower bound: -12.3205981, upper bound: 12.3205985
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 8, lower bound: -12.3205981, upper bound: 12.3205985
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 8, lower bound: -12.3208029, upper bound: 12.3208030
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 8, lower bound: -12.3208029, upper bound: 12.3208030
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 8, lower bound: -12.3208012, upper bound: 12.3207990
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 8, lower bound: -12.3207990, upper bound: 12.3208006
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 8, lower bound: -12.3206156, upper bound: 12.3206160
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 8, lower bound: -12.3206156, upper bound: 12.3206160
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 8, lower bound: -12.3209052, upper bound: 12.3209052
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 8, lower bound: -12.3209052, upper bound: 12.3209047
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 8, lower bound: -12.3208510, upper bound: 12.3208503
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 8, lower bound: -12.3208501, upper bound: 12.3208502
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 8, lower bound: -12.3208395, upper bound: 12.3208364
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 8, lower bound: -12.3208366, upper bound: 12.3208391
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 8, lower bound: -12.3204312, upper bound: 12.3204310
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 8, lower bound: -12.3204311, upper bound: 12.3204312
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 8, lower bound: -12.3204996, upper bound: 12.3204992
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 8, lower bound: -12.3204994, upper bound: 12.3204996
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 8, lower bound: -12.3204678, upper bound: 12.3204678
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 8, lower bound: -12.3204678, upper bound: 12.3204678
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 8, lower bound: -12.3208424, upper bound: 12.3208423
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 8.99
Output dim: 8, lower bound: -12.3208424, upper bound: 12.3208423

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 6.56 + 594.84 = 601.40 seconds
