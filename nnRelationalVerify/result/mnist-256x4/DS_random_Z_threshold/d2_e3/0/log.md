## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 12.3086572218


## IAR start

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
execution time: IAR + RelationalAnalysis = 0.72 + 4.98 = 5.70 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -12.3209782, upper bound: 12.3209782

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3206470, upper bound: 12.3206458
time: 3.12 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3206470, upper bound: 12.3206458
time: 3.13 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 6.27 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 6.27
Output dim: 8, lower bound: -12.3206470, upper bound: 12.3206458
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 6.27
Output dim: 8, lower bound: -12.3206470, upper bound: 12.3206458

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 191

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205291, upper bound: 12.3205291
time: 2.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205291, upper bound: 12.3205291
time: 2.71 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3206463, upper bound: 12.3206467
time: 2.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3206470, upper bound: 12.3206470
time: 3.09 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 6.60 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 6.60
Output dim: 8, lower bound: -12.3205291, upper bound: 12.3205291
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 6.60
Output dim: 8, lower bound: -12.3205291, upper bound: 12.3205291
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 6.60
Output dim: 8, lower bound: -12.3206463, upper bound: 12.3206467
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 6.60
Output dim: 8, lower bound: -12.3206470, upper bound: 12.3206470

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 210

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204635, upper bound: 12.3204635
time: 3.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204635, upper bound: 12.3204635
time: 3.01 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 92

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205290, upper bound: 12.3205291
time: 3.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205291, upper bound: 12.3205290
time: 3.40 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 219

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3206470, upper bound: 12.3206460
time: 3.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3206470, upper bound: 12.3206462
time: 2.90 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 88

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204098, upper bound: 12.3204097
time: 2.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204098, upper bound: 12.3204096
time: 2.50 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 5.83 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 5.83
Output dim: 8, lower bound: -12.3204635, upper bound: 12.3204635
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 5.83
Output dim: 8, lower bound: -12.3204635, upper bound: 12.3204635
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 5.83
Output dim: 8, lower bound: -12.3205290, upper bound: 12.3205291
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 5.83
Output dim: 8, lower bound: -12.3205291, upper bound: 12.3205290
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 5.83
Output dim: 8, lower bound: -12.3206470, upper bound: 12.3206460
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 5.83
Output dim: 8, lower bound: -12.3206470, upper bound: 12.3206462
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 5.83
Output dim: 8, lower bound: -12.3204098, upper bound: 12.3204097
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 5.83
Output dim: 8, lower bound: -12.3204098, upper bound: 12.3204096

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3186796, upper bound: 12.3186791
time: 3.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3186796, upper bound: 12.3186791
time: 2.56 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203839, upper bound: 12.3203837
time: 3.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203836, upper bound: 12.3203839
time: 2.76 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 134

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204883, upper bound: 12.3204883
time: 2.00 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204883, upper bound: 12.3204883
time: 2.31 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204640, upper bound: 12.3204638
time: 3.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204639, upper bound: 12.3204640
time: 3.01 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205936, upper bound: 12.3205928
time: 3.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205932, upper bound: 12.3205932
time: 3.21 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 188

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3206161, upper bound: 12.3206151
time: 2.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3206151, upper bound: 12.3206169
time: 5.09 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 62

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 109

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3201598, upper bound: 12.3201597
time: 9.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3201599, upper bound: 12.3201597
time: 2.45 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 188

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3200629, upper bound: 12.3200629
time: 1.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3200629, upper bound: 12.3200629
time: 1.96 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 4.64 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.64
Output dim: 8, lower bound: -12.3186796, upper bound: 12.3186791
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.64
Output dim: 8, lower bound: -12.3186796, upper bound: 12.3186791
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.64
Output dim: 8, lower bound: -12.3203839, upper bound: 12.3203837
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.64
Output dim: 8, lower bound: -12.3203836, upper bound: 12.3203839
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.64
Output dim: 8, lower bound: -12.3204883, upper bound: 12.3204883
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.64
Output dim: 8, lower bound: -12.3204883, upper bound: 12.3204883
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.64
Output dim: 8, lower bound: -12.3204640, upper bound: 12.3204638
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.64
Output dim: 8, lower bound: -12.3204639, upper bound: 12.3204640
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.64
Output dim: 8, lower bound: -12.3205936, upper bound: 12.3205928
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.64
Output dim: 8, lower bound: -12.3205932, upper bound: 12.3205932
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.64
Output dim: 8, lower bound: -12.3206161, upper bound: 12.3206151
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.64
Output dim: 8, lower bound: -12.3206151, upper bound: 12.3206169
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.64
Output dim: 8, lower bound: -12.3201598, upper bound: 12.3201597
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.64
Output dim: 8, lower bound: -12.3201599, upper bound: 12.3201597
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.64
Output dim: 8, lower bound: -12.3200629, upper bound: 12.3200629
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.64
Output dim: 8, lower bound: -12.3200629, upper bound: 12.3200629

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 92

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3186796, upper bound: 12.3186796
time: 2.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3186796, upper bound: 12.3186791
time: 2.00 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3167209, upper bound: 12.3167209
time: 1.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3167209, upper bound: 12.3167209
time: 1.58 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 188

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3201790, upper bound: 12.3201790
time: 2.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3201790, upper bound: 12.3201790
time: 2.75 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3200029, upper bound: 12.3200035
time: 2.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3200029, upper bound: 12.3200035
time: 2.45 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 169

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204882, upper bound: 12.3204883
time: 3.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204883, upper bound: 12.3204882
time: 2.67 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 169

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204883, upper bound: 12.3204881
time: 2.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204881, upper bound: 12.3204883
time: 2.90 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204639, upper bound: 12.3204638
time: 2.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204640, upper bound: 12.3204638
time: 3.36 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204129, upper bound: 12.3204135
time: 2.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204129, upper bound: 12.3204135
time: 2.59 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205930, upper bound: 12.3205927
time: 2.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205934, upper bound: 12.3205929
time: 4.16 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203462, upper bound: 12.3203470
time: 3.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203462, upper bound: 12.3203470
time: 3.65 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 109

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204684, upper bound: 12.3204678
time: 3.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204682, upper bound: 12.3204677
time: 3.38 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205609, upper bound: 12.3205620
time: 3.08 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205609, upper bound: 12.3205621
time: 3.11 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3199722, upper bound: 12.3199721
time: 2.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3199722, upper bound: 12.3199721
time: 2.42 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 72

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3201489, upper bound: 12.3201479
time: 2.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3201479, upper bound: 12.3201487
time: 1.66 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 191

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3198060, upper bound: 12.3198060
time: 2.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3198060, upper bound: 12.3198060
time: 2.87 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 52

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 219

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3200629, upper bound: 12.3200627
time: 3.05 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3200628, upper bound: 12.3200629
time: 2.96 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 8.54 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 8.54
Output dim: 8, lower bound: -12.3186796, upper bound: 12.3186796
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 8.54
Output dim: 8, lower bound: -12.3186796, upper bound: 12.3186791
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 8.54
Output dim: 8, lower bound: -12.3167209, upper bound: 12.3167209
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 8.54
Output dim: 8, lower bound: -12.3167209, upper bound: 12.3167209
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 8.54
Output dim: 8, lower bound: -12.3201790, upper bound: 12.3201790
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 8.54
Output dim: 8, lower bound: -12.3201790, upper bound: 12.3201790
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 8.54
Output dim: 8, lower bound: -12.3200029, upper bound: 12.3200035
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 8.54
Output dim: 8, lower bound: -12.3200029, upper bound: 12.3200035
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 8.54
Output dim: 8, lower bound: -12.3204882, upper bound: 12.3204883
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 8.54
Output dim: 8, lower bound: -12.3204883, upper bound: 12.3204882
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 8.54
Output dim: 8, lower bound: -12.3204883, upper bound: 12.3204881
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 8.54
Output dim: 8, lower bound: -12.3204881, upper bound: 12.3204883
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 8.54
Output dim: 8, lower bound: -12.3204639, upper bound: 12.3204638
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 8.54
Output dim: 8, lower bound: -12.3204640, upper bound: 12.3204638
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 8.54
Output dim: 8, lower bound: -12.3204129, upper bound: 12.3204135
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 8.54
Output dim: 8, lower bound: -12.3204129, upper bound: 12.3204135
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 8.54
Output dim: 8, lower bound: -12.3205930, upper bound: 12.3205927
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 8.54
Output dim: 8, lower bound: -12.3205934, upper bound: 12.3205929
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 8.54
Output dim: 8, lower bound: -12.3203462, upper bound: 12.3203470
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 8.54
Output dim: 8, lower bound: -12.3203462, upper bound: 12.3203470
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 8.54
Output dim: 8, lower bound: -12.3204684, upper bound: 12.3204678
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 8.54
Output dim: 8, lower bound: -12.3204682, upper bound: 12.3204677
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 8.54
Output dim: 8, lower bound: -12.3205609, upper bound: 12.3205620
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 8.54
Output dim: 8, lower bound: -12.3205609, upper bound: 12.3205621
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 8.54
Output dim: 8, lower bound: -12.3199722, upper bound: 12.3199721
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 8.54
Output dim: 8, lower bound: -12.3199722, upper bound: 12.3199721
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 8.54
Output dim: 8, lower bound: -12.3201489, upper bound: 12.3201479
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 8.54
Output dim: 8, lower bound: -12.3201479, upper bound: 12.3201487
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 8.54
Output dim: 8, lower bound: -12.3198060, upper bound: 12.3198060
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 8.54
Output dim: 8, lower bound: -12.3198060, upper bound: 12.3198060
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 8.54
Output dim: 8, lower bound: -12.3200629, upper bound: 12.3200627
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 8.54
Output dim: 8, lower bound: -12.3200628, upper bound: 12.3200629

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 72

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3186796, upper bound: 12.3186796
time: 1.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3186796, upper bound: 12.3186796
time: 2.00 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 169

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 78

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 188

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 107

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3169950, upper bound: 12.3169950
time: 2.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3169950, upper bound: 12.3169950
time: 1.46 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 72

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 78

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3167147, upper bound: 12.3167147
time: 1.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3167147, upper bound: 12.3167147
time: 1.51 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 62

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 92

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3167209, upper bound: 12.3167209
time: 1.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3167209, upper bound: 12.3167209
time: 1.82 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3201272, upper bound: 12.3201270
time: 2.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3201272, upper bound: 12.3201270
time: 2.93 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 72

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3201677, upper bound: 12.3201675
time: 3.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3201678, upper bound: 12.3201675
time: 2.78 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3199959, upper bound: 12.3199966
time: 3.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3199960, upper bound: 12.3199964
time: 3.26 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 78

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3196024, upper bound: 12.3196028
time: 2.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3196024, upper bound: 12.3196029
time: 3.09 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 88

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 62

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 107

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204762, upper bound: 12.3204743
time: 2.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204743, upper bound: 12.3204762
time: 3.35 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 88

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204818, upper bound: 12.3204819
time: 2.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204819, upper bound: 12.3204817
time: 2.77 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204820, upper bound: 12.3204820
time: 3.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204819, upper bound: 12.3204816
time: 2.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204818, upper bound: 12.3204820
time: 3.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204820, upper bound: 12.3204819
time: 3.01 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 210

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203837, upper bound: 12.3203836
time: 2.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203836, upper bound: 12.3203838
time: 2.13 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 72

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 109

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3202195, upper bound: 12.3202190
time: 3.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3202194, upper bound: 12.3202191
time: 3.28 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Candidate
type: DSZ, layer: 1, pos: 210

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203017, upper bound: 12.3203022
time: 3.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203017, upper bound: 12.3203024
time: 3.54 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 72

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 195

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 175

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 169

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204129, upper bound: 12.3204132
time: 3.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204127, upper bound: 12.3204135
time: 2.88 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 109

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204393, upper bound: 12.3204395
time: 3.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204393, upper bound: 12.3204395
time: 2.44 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204328, upper bound: 12.3204323
time: 1.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204328, upper bound: 12.3204323
time: 1.68 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3193139, upper bound: 12.3193134
time: 2.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3193139, upper bound: 12.3193134
time: 1.85 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 78

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3200352, upper bound: 12.3200354
time: 7.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3200351, upper bound: 12.3200353
time: 2.48 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 72

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 169

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204673, upper bound: 12.3204666
time: 2.90 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204666, upper bound: 12.3204667
time: 2.80 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204211, upper bound: 12.3204203
time: 2.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204211, upper bound: 12.3204202
time: 2.54 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 134

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205119, upper bound: 12.3205133
time: 3.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205119, upper bound: 12.3205133
time: 3.89 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 175

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3192273, upper bound: 12.3192276
time: 2.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3192274, upper bound: 12.3192276
time: 3.30 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3199722, upper bound: 12.3199717
time: 3.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3199718, upper bound: 12.3199721
time: 2.92 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 188

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 175

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3195782, upper bound: 12.3195776
time: 2.34 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3195778, upper bound: 12.3195780
time: 2.08 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3201479, upper bound: 12.3201475
time: 7.11 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3201486, upper bound: 12.3201477
time: 3.04 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 188

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3184947, upper bound: 12.3184946
time: 3.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3184947, upper bound: 12.3184946
time: 3.76 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 169

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 219

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3198060, upper bound: 12.3198059
time: 3.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3198060, upper bound: 12.3198060
time: 2.90 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 52

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 191

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 134

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3185213, upper bound: 12.3185214
time: 2.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3185213, upper bound: 12.3185213
time: 2.15 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 62

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 169

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3200629, upper bound: 12.3200628
time: 2.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3200629, upper bound: 12.3200628
time: 2.57 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 134

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3192077, upper bound: 12.3192073
time: 1.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3192077, upper bound: 12.3192073
time: 2.98 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 5.61 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 8, lower bound: -12.3186796, upper bound: 12.3186796
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 8, lower bound: -12.3186796, upper bound: 12.3186796
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 8, lower bound: -12.3169950, upper bound: 12.3169950
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 8, lower bound: -12.3169950, upper bound: 12.3169950
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 8, lower bound: -12.3167147, upper bound: 12.3167147
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 8, lower bound: -12.3167147, upper bound: 12.3167147
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 8, lower bound: -12.3167209, upper bound: 12.3167209
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 8, lower bound: -12.3167209, upper bound: 12.3167209
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 8, lower bound: -12.3201272, upper bound: 12.3201270
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 8, lower bound: -12.3201272, upper bound: 12.3201270
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 8, lower bound: -12.3201677, upper bound: 12.3201675
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 8, lower bound: -12.3201678, upper bound: 12.3201675
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 8, lower bound: -12.3199959, upper bound: 12.3199966
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 8, lower bound: -12.3199960, upper bound: 12.3199964
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 8, lower bound: -12.3196024, upper bound: 12.3196028
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 8, lower bound: -12.3196024, upper bound: 12.3196029
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 8, lower bound: -12.3204762, upper bound: 12.3204743
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 8, lower bound: -12.3204743, upper bound: 12.3204762
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 8, lower bound: -12.3204818, upper bound: 12.3204819
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 8, lower bound: -12.3204819, upper bound: 12.3204817
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 8, lower bound: -12.3204820, upper bound: 12.3204820
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 8, lower bound: -12.3204819, upper bound: 12.3204816
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 8, lower bound: -12.3204818, upper bound: 12.3204820
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 8, lower bound: -12.3204820, upper bound: 12.3204819
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 8, lower bound: -12.3203837, upper bound: 12.3203836
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 8, lower bound: -12.3203836, upper bound: 12.3203838
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 8, lower bound: -12.3202195, upper bound: 12.3202190
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 8, lower bound: -12.3202194, upper bound: 12.3202191
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 8, lower bound: -12.3203017, upper bound: 12.3203022
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 8, lower bound: -12.3203017, upper bound: 12.3203024
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 8, lower bound: -12.3204129, upper bound: 12.3204132
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 8, lower bound: -12.3204127, upper bound: 12.3204135
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 8, lower bound: -12.3204393, upper bound: 12.3204395
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 8, lower bound: -12.3204393, upper bound: 12.3204395
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 8, lower bound: -12.3204328, upper bound: 12.3204323
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 8, lower bound: -12.3204328, upper bound: 12.3204323
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 8, lower bound: -12.3193139, upper bound: 12.3193134
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 8, lower bound: -12.3193139, upper bound: 12.3193134
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 8, lower bound: -12.3200352, upper bound: 12.3200354
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 8, lower bound: -12.3200351, upper bound: 12.3200353
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 8, lower bound: -12.3204673, upper bound: 12.3204666
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 8, lower bound: -12.3204666, upper bound: 12.3204667
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 8, lower bound: -12.3204211, upper bound: 12.3204203
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 8, lower bound: -12.3204211, upper bound: 12.3204202
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 8, lower bound: -12.3205119, upper bound: 12.3205133
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 8, lower bound: -12.3205119, upper bound: 12.3205133
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 8, lower bound: -12.3192273, upper bound: 12.3192276
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 8, lower bound: -12.3192274, upper bound: 12.3192276
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 8, lower bound: -12.3199722, upper bound: 12.3199717
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 8, lower bound: -12.3199718, upper bound: 12.3199721
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 8, lower bound: -12.3195782, upper bound: 12.3195776
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 8, lower bound: -12.3195778, upper bound: 12.3195780
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 8, lower bound: -12.3201479, upper bound: 12.3201475
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 8, lower bound: -12.3201486, upper bound: 12.3201477
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 8, lower bound: -12.3184947, upper bound: 12.3184946
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 8, lower bound: -12.3184947, upper bound: 12.3184946
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 8, lower bound: -12.3198060, upper bound: 12.3198059
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 8, lower bound: -12.3198060, upper bound: 12.3198060
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 8, lower bound: -12.3185213, upper bound: 12.3185214
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 8, lower bound: -12.3185213, upper bound: 12.3185213
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 8, lower bound: -12.3200629, upper bound: 12.3200628
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 8, lower bound: -12.3200629, upper bound: 12.3200628
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 8, lower bound: -12.3192077, upper bound: 12.3192073
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 8, lower bound: -12.3192077, upper bound: 12.3192073

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 219

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3186796, upper bound: 12.3186796
time: 1.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3186796, upper bound: 12.3186796
time: 7.19 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3185951, upper bound: 12.3185951
time: 2.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3185951, upper bound: 12.3185951
time: 8.17 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3132891, upper bound: 12.3132891
time: 1.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3132891, upper bound: 12.3132891
time: 1.56 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 175

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 219

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3169950, upper bound: 12.3169950
time: 1.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3169950, upper bound: 12.3169950
time: 1.79 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 52

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3129835, upper bound: 12.3129835
time: 2.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3129835, upper bound: 12.3129835
time: 1.63 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 169

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3167147, upper bound: 12.3167147
time: 1.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3167147, upper bound: 12.3167147
time: 1.39 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 88

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3115350, upper bound: 12.3115350
time: 1.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3115350, upper bound: 12.3115350
time: 1.84 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 52

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 78

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3167147, upper bound: 12.3167147
time: 1.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3167147, upper bound: 12.3167147
time: 1.62 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 134

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3200328, upper bound: 12.3200333
time: 3.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3200339, upper bound: 12.3200333
time: 2.81 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 52

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3199288, upper bound: 12.3199283
time: 3.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3199286, upper bound: 12.3199287
time: 2.00 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 78

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3199684, upper bound: 12.3199683
time: 3.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3199684, upper bound: 12.3199683
time: 4.14 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3201609, upper bound: 12.3201608
time: 3.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3201610, upper bound: 12.3201608
time: 2.83 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 85

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3199958, upper bound: 12.3199953
time: 2.05 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3199948, upper bound: 12.3199965
time: 1.88 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 72

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3199666, upper bound: 12.3199660
time: 2.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3199659, upper bound: 12.3199667
time: 1.81 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 109

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3180577, upper bound: 12.3180580
time: 2.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3180577, upper bound: 12.3180580
time: 2.52 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3195780, upper bound: 12.3195783
time: 3.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3195777, upper bound: 12.3195788
time: 2.15 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 169

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 88

### Candidate
type: DSZ, layer: 1, pos: 175

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204157, upper bound: 12.3204132
time: 2.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204149, upper bound: 12.3204141
time: 3.32 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203746, upper bound: 12.3203760
time: 2.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203744, upper bound: 12.3203765
time: 2.58 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 5.99 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.99
Output dim: 8, lower bound: -12.3186796, upper bound: 12.3186796
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.99
Output dim: 8, lower bound: -12.3186796, upper bound: 12.3186796
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.99
Output dim: 8, lower bound: -12.3185951, upper bound: 12.3185951
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.99
Output dim: 8, lower bound: -12.3185951, upper bound: 12.3185951
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.99
Output dim: 8, lower bound: -12.3132891, upper bound: 12.3132891
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.99
Output dim: 8, lower bound: -12.3132891, upper bound: 12.3132891
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.99
Output dim: 8, lower bound: -12.3169950, upper bound: 12.3169950
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.99
Output dim: 8, lower bound: -12.3169950, upper bound: 12.3169950
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.99
Output dim: 8, lower bound: -12.3129835, upper bound: 12.3129835
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.99
Output dim: 8, lower bound: -12.3129835, upper bound: 12.3129835
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.99
Output dim: 8, lower bound: -12.3167147, upper bound: 12.3167147
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.99
Output dim: 8, lower bound: -12.3167147, upper bound: 12.3167147
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.99
Output dim: 8, lower bound: -12.3115350, upper bound: 12.3115350
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.99
Output dim: 8, lower bound: -12.3115350, upper bound: 12.3115350
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.99
Output dim: 8, lower bound: -12.3167147, upper bound: 12.3167147
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.99
Output dim: 8, lower bound: -12.3167147, upper bound: 12.3167147
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.99
Output dim: 8, lower bound: -12.3200328, upper bound: 12.3200333
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.99
Output dim: 8, lower bound: -12.3200339, upper bound: 12.3200333
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.99
Output dim: 8, lower bound: -12.3199288, upper bound: 12.3199283
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.99
Output dim: 8, lower bound: -12.3199286, upper bound: 12.3199287
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.99
Output dim: 8, lower bound: -12.3199684, upper bound: 12.3199683
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.99
Output dim: 8, lower bound: -12.3199684, upper bound: 12.3199683
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.99
Output dim: 8, lower bound: -12.3201609, upper bound: 12.3201608
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.99
Output dim: 8, lower bound: -12.3201610, upper bound: 12.3201608
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.99
Output dim: 8, lower bound: -12.3199958, upper bound: 12.3199953
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.99
Output dim: 8, lower bound: -12.3199948, upper bound: 12.3199965
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.99
Output dim: 8, lower bound: -12.3199666, upper bound: 12.3199660
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.99
Output dim: 8, lower bound: -12.3199659, upper bound: 12.3199667
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.99
Output dim: 8, lower bound: -12.3180577, upper bound: 12.3180580
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.99
Output dim: 8, lower bound: -12.3180577, upper bound: 12.3180580
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.99
Output dim: 8, lower bound: -12.3195780, upper bound: 12.3195783
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.99
Output dim: 8, lower bound: -12.3195777, upper bound: 12.3195788
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.99
Output dim: 8, lower bound: -12.3204157, upper bound: 12.3204132
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.99
Output dim: 8, lower bound: -12.3204149, upper bound: 12.3204141
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.99
Output dim: 8, lower bound: -12.3203746, upper bound: 12.3203760
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.99
Output dim: 8, lower bound: -12.3203744, upper bound: 12.3203765
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 8, lower bound: -12.3204818, upper bound: 12.3204819
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 8, lower bound: -12.3204819, upper bound: 12.3204817
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 8, lower bound: -12.3204820, upper bound: 12.3204820
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 8, lower bound: -12.3204819, upper bound: 12.3204816
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 8, lower bound: -12.3204818, upper bound: 12.3204820
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 8, lower bound: -12.3204820, upper bound: 12.3204819
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 8, lower bound: -12.3203837, upper bound: 12.3203836
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 8, lower bound: -12.3203836, upper bound: 12.3203838
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 8, lower bound: -12.3202195, upper bound: 12.3202190
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 8, lower bound: -12.3202194, upper bound: 12.3202191
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 8, lower bound: -12.3203017, upper bound: 12.3203022
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 8, lower bound: -12.3203017, upper bound: 12.3203024
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 8, lower bound: -12.3204129, upper bound: 12.3204132
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 8, lower bound: -12.3204127, upper bound: 12.3204135
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 8, lower bound: -12.3204393, upper bound: 12.3204395
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 8, lower bound: -12.3204393, upper bound: 12.3204395
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 8, lower bound: -12.3204328, upper bound: 12.3204323
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 8, lower bound: -12.3204328, upper bound: 12.3204323
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 8, lower bound: -12.3193139, upper bound: 12.3193134
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 8, lower bound: -12.3193139, upper bound: 12.3193134
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 8, lower bound: -12.3200352, upper bound: 12.3200354
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 8, lower bound: -12.3200351, upper bound: 12.3200353
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 8, lower bound: -12.3204673, upper bound: 12.3204666
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 8, lower bound: -12.3204666, upper bound: 12.3204667
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 8, lower bound: -12.3204211, upper bound: 12.3204203
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 8, lower bound: -12.3204211, upper bound: 12.3204202
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 8, lower bound: -12.3205119, upper bound: 12.3205133
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 8, lower bound: -12.3205119, upper bound: 12.3205133
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 8, lower bound: -12.3192273, upper bound: 12.3192276
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 8, lower bound: -12.3192274, upper bound: 12.3192276
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 8, lower bound: -12.3199722, upper bound: 12.3199717
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 8, lower bound: -12.3199718, upper bound: 12.3199721
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 8, lower bound: -12.3195782, upper bound: 12.3195776
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 8, lower bound: -12.3195778, upper bound: 12.3195780
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 8, lower bound: -12.3201479, upper bound: 12.3201475
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 8, lower bound: -12.3201486, upper bound: 12.3201477
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 8, lower bound: -12.3184947, upper bound: 12.3184946
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 8, lower bound: -12.3184947, upper bound: 12.3184946
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 8, lower bound: -12.3198060, upper bound: 12.3198059
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 8, lower bound: -12.3198060, upper bound: 12.3198060
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 8, lower bound: -12.3185213, upper bound: 12.3185214
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 8, lower bound: -12.3185213, upper bound: 12.3185213
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 8, lower bound: -12.3200629, upper bound: 12.3200628
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 8, lower bound: -12.3200629, upper bound: 12.3200628
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 8, lower bound: -12.3192077, upper bound: 12.3192073
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 8, lower bound: -12.3192077, upper bound: 12.3192073

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 5.70 + 597.07 = 602.77 seconds
