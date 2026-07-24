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
execution time: IAR + RelationalAnalysis = 1.72 + 5.14 = 6.87 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -12.3209782, upper bound: 12.3209782

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3209462, upper bound: 12.3209459
time: 4.46 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3209459, upper bound: 12.3209462
time: 4.98 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 9.60 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 9.60
Output dim: 8, lower bound: -12.3209462, upper bound: 12.3209459
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 9.60
Output dim: 8, lower bound: -12.3209459, upper bound: 12.3209462

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

Time for backsubstitution: 1.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3206969, upper bound: 12.3206969
time: 4.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3206969, upper bound: 12.3206963
time: 3.02 seconds

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

Time for backsubstitution: 1.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3206969, upper bound: 12.3206969
time: 3.97 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3206969, upper bound: 12.3206962
time: 4.58 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 10.21 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 10.21
Output dim: 8, lower bound: -12.3206969, upper bound: 12.3206969
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 10.21
Output dim: 8, lower bound: -12.3206969, upper bound: 12.3206963
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 10.21
Output dim: 8, lower bound: -12.3206969, upper bound: 12.3206969
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 10.21
Output dim: 8, lower bound: -12.3206969, upper bound: 12.3206962

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

Time for backsubstitution: 1.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 109

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205418, upper bound: 12.3205409
time: 4.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205409, upper bound: 12.3205414
time: 3.54 seconds

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

Time for backsubstitution: 1.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 109

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205418, upper bound: 12.3205410
time: 2.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205409, upper bound: 12.3205414
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

Time for backsubstitution: 1.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 109

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205413, upper bound: 12.3205414
time: 3.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205409, upper bound: 12.3205415
time: 3.86 seconds

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

Time for backsubstitution: 1.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 109

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205413, upper bound: 12.3205414
time: 3.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3205409, upper bound: 12.3205415
time: 3.94 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 9.31 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 9.31
Output dim: 8, lower bound: -12.3205418, upper bound: 12.3205409
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 9.31
Output dim: 8, lower bound: -12.3205409, upper bound: 12.3205414
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 9.31
Output dim: 8, lower bound: -12.3205418, upper bound: 12.3205410
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 9.31
Output dim: 8, lower bound: -12.3205409, upper bound: 12.3205414
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 9.31
Output dim: 8, lower bound: -12.3205413, upper bound: 12.3205414
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 9.31
Output dim: 8, lower bound: -12.3205409, upper bound: 12.3205415
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 9.31
Output dim: 8, lower bound: -12.3205413, upper bound: 12.3205414
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 9.31
Output dim: 8, lower bound: -12.3205409, upper bound: 12.3205415

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

Time for backsubstitution: 1.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204811, upper bound: 12.3204803
time: 2.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204805, upper bound: 12.3204814
time: 2.60 seconds

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

Time for backsubstitution: 1.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204809, upper bound: 12.3204804
time: 2.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204801, upper bound: 12.3204811
time: 2.38 seconds

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

Time for backsubstitution: 1.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204811, upper bound: 12.3204803
time: 2.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204805, upper bound: 12.3204814
time: 2.55 seconds

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

Time for backsubstitution: 1.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204809, upper bound: 12.3204804
time: 3.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204801, upper bound: 12.3204816
time: 2.57 seconds

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

Time for backsubstitution: 1.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204811, upper bound: 12.3204803
time: 2.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204805, upper bound: 12.3204814
time: 3.37 seconds

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

Time for backsubstitution: 1.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204814, upper bound: 12.3204804
time: 2.97 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204801, upper bound: 12.3204816
time: 2.92 seconds

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

Time for backsubstitution: 1.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204811, upper bound: 12.3204803
time: 2.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204803, upper bound: 12.3204814
time: 3.51 seconds

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

Time for backsubstitution: 1.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204814, upper bound: 12.3204804
time: 2.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204801, upper bound: 12.3204816
time: 2.92 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 7.57 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 7.57
Output dim: 8, lower bound: -12.3204811, upper bound: 12.3204803
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 7.57
Output dim: 8, lower bound: -12.3204805, upper bound: 12.3204814
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 7.57
Output dim: 8, lower bound: -12.3204809, upper bound: 12.3204804
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 7.57
Output dim: 8, lower bound: -12.3204801, upper bound: 12.3204811
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 7.57
Output dim: 8, lower bound: -12.3204811, upper bound: 12.3204803
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 7.57
Output dim: 8, lower bound: -12.3204805, upper bound: 12.3204814
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 7.57
Output dim: 8, lower bound: -12.3204809, upper bound: 12.3204804
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 7.57
Output dim: 8, lower bound: -12.3204801, upper bound: 12.3204816
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 7.57
Output dim: 8, lower bound: -12.3204811, upper bound: 12.3204803
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 7.57
Output dim: 8, lower bound: -12.3204805, upper bound: 12.3204814
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 7.57
Output dim: 8, lower bound: -12.3204814, upper bound: 12.3204804
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 7.57
Output dim: 8, lower bound: -12.3204801, upper bound: 12.3204816
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 7.57
Output dim: 8, lower bound: -12.3204811, upper bound: 12.3204803
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 7.57
Output dim: 8, lower bound: -12.3204803, upper bound: 12.3204814
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 7.57
Output dim: 8, lower bound: -12.3204814, upper bound: 12.3204804
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 7.57
Output dim: 8, lower bound: -12.3204801, upper bound: 12.3204816

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

Time for backsubstitution: 1.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 92

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204816, upper bound: 12.3204800
time: 2.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204816, upper bound: 12.3204803
time: 2.58 seconds

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

Time for backsubstitution: 1.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 92

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204803, upper bound: 12.3204811
time: 2.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204799, upper bound: 12.3204814
time: 2.73 seconds

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

Time for backsubstitution: 1.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 92

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204809, upper bound: 12.3204804
time: 2.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204814, upper bound: 12.3204804
time: 3.76 seconds

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

Time for backsubstitution: 1.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 92

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204803, upper bound: 12.3204816
time: 4.09 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204803, upper bound: 12.3204816
time: 2.55 seconds

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

Time for backsubstitution: 1.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 92

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204816, upper bound: 12.3204800
time: 2.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204816, upper bound: 12.3204803
time: 2.62 seconds

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

Time for backsubstitution: 1.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 92

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204804, upper bound: 12.3204809
time: 3.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204799, upper bound: 12.3204814
time: 2.75 seconds

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

Time for backsubstitution: 1.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 92

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204809, upper bound: 12.3204804
time: 2.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204803, upper bound: 12.3204805
time: 2.91 seconds

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

Time for backsubstitution: 1.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 92

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204803, upper bound: 12.3204816
time: 3.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204803, upper bound: 12.3204813
time: 2.52 seconds

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

Time for backsubstitution: 1.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 92

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204816, upper bound: 12.3204802
time: 3.00 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204816, upper bound: 12.3204803
time: 2.22 seconds

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

Time for backsubstitution: 1.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 92

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204805, upper bound: 12.3204814
time: 4.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204799, upper bound: 12.3204814
time: 2.32 seconds

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

Time for backsubstitution: 1.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 92

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204809, upper bound: 12.3204804
time: 3.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204814, upper bound: 12.3204804
time: 2.73 seconds

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

Time for backsubstitution: 1.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 92

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204803, upper bound: 12.3204816
time: 4.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204803, upper bound: 12.3204816
time: 2.26 seconds

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

Time for backsubstitution: 1.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 92

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204816, upper bound: 12.3204802
time: 3.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204816, upper bound: 12.3204803
time: 2.27 seconds

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

Time for backsubstitution: 1.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 92

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204800, upper bound: 12.3204814
time: 3.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204799, upper bound: 12.3204814
time: 2.25 seconds

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

Time for backsubstitution: 1.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 92

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204809, upper bound: 12.3204804
time: 3.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204814, upper bound: 12.3204804
time: 2.73 seconds

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

Time for backsubstitution: 1.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 92

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204803, upper bound: 12.3204816
time: 3.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3204803, upper bound: 12.3204811
time: 2.28 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 9.74 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 9.74
Output dim: 8, lower bound: -12.3204816, upper bound: 12.3204800
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 9.74
Output dim: 8, lower bound: -12.3204816, upper bound: 12.3204803
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 9.74
Output dim: 8, lower bound: -12.3204803, upper bound: 12.3204811
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 9.74
Output dim: 8, lower bound: -12.3204799, upper bound: 12.3204814
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 9.74
Output dim: 8, lower bound: -12.3204809, upper bound: 12.3204804
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 9.74
Output dim: 8, lower bound: -12.3204814, upper bound: 12.3204804
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 9.74
Output dim: 8, lower bound: -12.3204803, upper bound: 12.3204816
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 9.74
Output dim: 8, lower bound: -12.3204803, upper bound: 12.3204816
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 9.74
Output dim: 8, lower bound: -12.3204816, upper bound: 12.3204800
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 9.74
Output dim: 8, lower bound: -12.3204816, upper bound: 12.3204803
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 9.74
Output dim: 8, lower bound: -12.3204804, upper bound: 12.3204809
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 9.74
Output dim: 8, lower bound: -12.3204799, upper bound: 12.3204814
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 9.74
Output dim: 8, lower bound: -12.3204809, upper bound: 12.3204804
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 9.74
Output dim: 8, lower bound: -12.3204803, upper bound: 12.3204805
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 9.74
Output dim: 8, lower bound: -12.3204803, upper bound: 12.3204816
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 9.74
Output dim: 8, lower bound: -12.3204803, upper bound: 12.3204813
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 9.74
Output dim: 8, lower bound: -12.3204816, upper bound: 12.3204802
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 9.74
Output dim: 8, lower bound: -12.3204816, upper bound: 12.3204803
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 9.74
Output dim: 8, lower bound: -12.3204805, upper bound: 12.3204814
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 9.74
Output dim: 8, lower bound: -12.3204799, upper bound: 12.3204814
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 9.74
Output dim: 8, lower bound: -12.3204809, upper bound: 12.3204804
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 9.74
Output dim: 8, lower bound: -12.3204814, upper bound: 12.3204804
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 9.74
Output dim: 8, lower bound: -12.3204803, upper bound: 12.3204816
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 9.74
Output dim: 8, lower bound: -12.3204803, upper bound: 12.3204816
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 9.74
Output dim: 8, lower bound: -12.3204816, upper bound: 12.3204802
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 9.74
Output dim: 8, lower bound: -12.3204816, upper bound: 12.3204803
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 9.74
Output dim: 8, lower bound: -12.3204800, upper bound: 12.3204814
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 9.74
Output dim: 8, lower bound: -12.3204799, upper bound: 12.3204814
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 9.74
Output dim: 8, lower bound: -12.3204809, upper bound: 12.3204804
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 9.74
Output dim: 8, lower bound: -12.3204814, upper bound: 12.3204804
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 9.74
Output dim: 8, lower bound: -12.3204803, upper bound: 12.3204816
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 9.74
Output dim: 8, lower bound: -12.3204803, upper bound: 12.3204811

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

Time for backsubstitution: 1.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203836, upper bound: 12.3203824
time: 4.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203833, upper bound: 12.3203828
time: 3.06 seconds

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

Time for backsubstitution: 1.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203836, upper bound: 12.3203824
time: 3.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203833, upper bound: 12.3203828
time: 7.21 seconds

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

Time for backsubstitution: 1.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203828, upper bound: 12.3203833
time: 3.13 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203824, upper bound: 12.3203836
time: 3.27 seconds

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

Time for backsubstitution: 1.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203828, upper bound: 12.3203833
time: 2.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203824, upper bound: 12.3203836
time: 3.19 seconds

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

Time for backsubstitution: 1.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203836, upper bound: 12.3203824
time: 3.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203833, upper bound: 12.3203828
time: 2.30 seconds

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

Time for backsubstitution: 1.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203836, upper bound: 12.3203824
time: 2.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203833, upper bound: 12.3203828
time: 4.45 seconds

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

Time for backsubstitution: 1.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203828, upper bound: 12.3203833
time: 2.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203824, upper bound: 12.3203836
time: 4.17 seconds

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

Time for backsubstitution: 1.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203828, upper bound: 12.3203833
time: 2.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203824, upper bound: 12.3203836
time: 4.41 seconds

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

Time for backsubstitution: 1.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203836, upper bound: 12.3203824
time: 3.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203833, upper bound: 12.3203829
time: 4.40 seconds

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

Time for backsubstitution: 1.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203836, upper bound: 12.3203824
time: 5.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203833, upper bound: 12.3203828
time: 3.37 seconds

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

Time for backsubstitution: 1.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203828, upper bound: 12.3203833
time: 4.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203824, upper bound: 12.3203836
time: 3.01 seconds

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

Time for backsubstitution: 1.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203828, upper bound: 12.3203833
time: 3.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203824, upper bound: 12.3203836
time: 3.23 seconds

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

Time for backsubstitution: 1.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203836, upper bound: 12.3203824
time: 3.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203833, upper bound: 12.3203828
time: 3.92 seconds

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

Time for backsubstitution: 1.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203836, upper bound: 12.3203824
time: 2.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203824, upper bound: 12.3203828
time: 3.89 seconds

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

Time for backsubstitution: 1.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203828, upper bound: 12.3203833
time: 2.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203824, upper bound: 12.3203836
time: 3.72 seconds

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

Time for backsubstitution: 1.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203828, upper bound: 12.3203833
time: 2.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203824, upper bound: 12.3203836
time: 5.17 seconds

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

Time for backsubstitution: 1.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203836, upper bound: 12.3203824
time: 3.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203833, upper bound: 12.3203828
time: 3.66 seconds

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

Time for backsubstitution: 1.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203836, upper bound: 12.3203824
time: 2.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203833, upper bound: 12.3203828
time: 4.49 seconds

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

Time for backsubstitution: 2.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203828, upper bound: 12.3203833
time: 3.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203824, upper bound: 12.3203836
time: 3.63 seconds

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

Time for backsubstitution: 1.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203828, upper bound: 12.3203833
time: 2.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203824, upper bound: 12.3203836
time: 3.23 seconds

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

Time for backsubstitution: 2.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203836, upper bound: 12.3203824
time: 3.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203833, upper bound: 12.3203828
time: 2.97 seconds

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

Time for backsubstitution: 1.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203836, upper bound: 12.3203824
time: 3.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203833, upper bound: 12.3203828
time: 4.58 seconds

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

Time for backsubstitution: 1.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203828, upper bound: 12.3203833
time: 4.00 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203824, upper bound: 12.3203836
time: 2.88 seconds

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

Time for backsubstitution: 1.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203829, upper bound: 12.3203833
time: 3.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203824, upper bound: 12.3203836
time: 3.09 seconds

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

Time for backsubstitution: 1.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203836, upper bound: 12.3203824
time: 3.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203833, upper bound: 12.3203828
time: 3.53 seconds

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

Time for backsubstitution: 1.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203836, upper bound: 12.3203824
time: 3.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203833, upper bound: 12.3203828
time: 4.47 seconds

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

Time for backsubstitution: 1.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203828, upper bound: 12.3203833
time: 3.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203824, upper bound: 12.3203836
time: 3.03 seconds

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

Time for backsubstitution: 1.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203828, upper bound: 12.3203833
time: 2.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203824, upper bound: 12.3203836
time: 3.33 seconds

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

Time for backsubstitution: 2.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203836, upper bound: 12.3203824
time: 3.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203833, upper bound: 12.3203828
time: 3.54 seconds

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

Time for backsubstitution: 2.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203836, upper bound: 12.3203824
time: 3.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203833, upper bound: 12.3203828
time: 3.63 seconds

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

Time for backsubstitution: 2.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203828, upper bound: 12.3203833
time: 3.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203824, upper bound: 12.3203836
time: 2.75 seconds

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

Time for backsubstitution: 1.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203828, upper bound: 12.3203833
time: 3.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3203824, upper bound: 12.3203836
time: 3.13 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 8.49 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 8.49
Output dim: 8, lower bound: -12.3203836, upper bound: 12.3203824
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 8.49
Output dim: 8, lower bound: -12.3203833, upper bound: 12.3203828
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 8.49
Output dim: 8, lower bound: -12.3203836, upper bound: 12.3203824
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 8.49
Output dim: 8, lower bound: -12.3203833, upper bound: 12.3203828
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 8.49
Output dim: 8, lower bound: -12.3203828, upper bound: 12.3203833
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 8.49
Output dim: 8, lower bound: -12.3203824, upper bound: 12.3203836
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 8.49
Output dim: 8, lower bound: -12.3203828, upper bound: 12.3203833
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 8.49
Output dim: 8, lower bound: -12.3203824, upper bound: 12.3203836
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 8.49
Output dim: 8, lower bound: -12.3203836, upper bound: 12.3203824
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 8.49
Output dim: 8, lower bound: -12.3203833, upper bound: 12.3203828
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 8.49
Output dim: 8, lower bound: -12.3203836, upper bound: 12.3203824
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 8.49
Output dim: 8, lower bound: -12.3203833, upper bound: 12.3203828
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 8.49
Output dim: 8, lower bound: -12.3203828, upper bound: 12.3203833
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 8.49
Output dim: 8, lower bound: -12.3203824, upper bound: 12.3203836
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 8.49
Output dim: 8, lower bound: -12.3203828, upper bound: 12.3203833
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 8.49
Output dim: 8, lower bound: -12.3203824, upper bound: 12.3203836
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 8.49
Output dim: 8, lower bound: -12.3203836, upper bound: 12.3203824
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 8.49
Output dim: 8, lower bound: -12.3203833, upper bound: 12.3203829
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 8.49
Output dim: 8, lower bound: -12.3203836, upper bound: 12.3203824
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 8.49
Output dim: 8, lower bound: -12.3203833, upper bound: 12.3203828
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 8.49
Output dim: 8, lower bound: -12.3203828, upper bound: 12.3203833
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 8.49
Output dim: 8, lower bound: -12.3203824, upper bound: 12.3203836
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 8.49
Output dim: 8, lower bound: -12.3203828, upper bound: 12.3203833
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 8.49
Output dim: 8, lower bound: -12.3203824, upper bound: 12.3203836
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 8.49
Output dim: 8, lower bound: -12.3203836, upper bound: 12.3203824
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 8.49
Output dim: 8, lower bound: -12.3203833, upper bound: 12.3203828
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 8.49
Output dim: 8, lower bound: -12.3203836, upper bound: 12.3203824
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 8.49
Output dim: 8, lower bound: -12.3203824, upper bound: 12.3203828
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 8.49
Output dim: 8, lower bound: -12.3203828, upper bound: 12.3203833
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 8.49
Output dim: 8, lower bound: -12.3203824, upper bound: 12.3203836
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 8.49
Output dim: 8, lower bound: -12.3203828, upper bound: 12.3203833
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 8.49
Output dim: 8, lower bound: -12.3203824, upper bound: 12.3203836
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 8.49
Output dim: 8, lower bound: -12.3203836, upper bound: 12.3203824
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 8.49
Output dim: 8, lower bound: -12.3203833, upper bound: 12.3203828
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 8.49
Output dim: 8, lower bound: -12.3203836, upper bound: 12.3203824
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 8.49
Output dim: 8, lower bound: -12.3203833, upper bound: 12.3203828
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 8.49
Output dim: 8, lower bound: -12.3203828, upper bound: 12.3203833
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 8.49
Output dim: 8, lower bound: -12.3203824, upper bound: 12.3203836
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 8.49
Output dim: 8, lower bound: -12.3203828, upper bound: 12.3203833
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 8.49
Output dim: 8, lower bound: -12.3203824, upper bound: 12.3203836
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 8.49
Output dim: 8, lower bound: -12.3203836, upper bound: 12.3203824
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 8.49
Output dim: 8, lower bound: -12.3203833, upper bound: 12.3203828
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 8.49
Output dim: 8, lower bound: -12.3203836, upper bound: 12.3203824
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 8.49
Output dim: 8, lower bound: -12.3203833, upper bound: 12.3203828
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 8.49
Output dim: 8, lower bound: -12.3203828, upper bound: 12.3203833
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 8.49
Output dim: 8, lower bound: -12.3203824, upper bound: 12.3203836
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 8.49
Output dim: 8, lower bound: -12.3203829, upper bound: 12.3203833
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 8.49
Output dim: 8, lower bound: -12.3203824, upper bound: 12.3203836
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 8.49
Output dim: 8, lower bound: -12.3203836, upper bound: 12.3203824
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 8.49
Output dim: 8, lower bound: -12.3203833, upper bound: 12.3203828
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 8.49
Output dim: 8, lower bound: -12.3203836, upper bound: 12.3203824
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 8.49
Output dim: 8, lower bound: -12.3203833, upper bound: 12.3203828
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 8.49
Output dim: 8, lower bound: -12.3203828, upper bound: 12.3203833
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 8.49
Output dim: 8, lower bound: -12.3203824, upper bound: 12.3203836
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 8.49
Output dim: 8, lower bound: -12.3203828, upper bound: 12.3203833
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 8.49
Output dim: 8, lower bound: -12.3203824, upper bound: 12.3203836
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 8.49
Output dim: 8, lower bound: -12.3203836, upper bound: 12.3203824
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 8.49
Output dim: 8, lower bound: -12.3203833, upper bound: 12.3203828
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 8.49
Output dim: 8, lower bound: -12.3203836, upper bound: 12.3203824
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 8.49
Output dim: 8, lower bound: -12.3203833, upper bound: 12.3203828
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 8.49
Output dim: 8, lower bound: -12.3203828, upper bound: 12.3203833
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 8.49
Output dim: 8, lower bound: -12.3203824, upper bound: 12.3203836
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 8.49
Output dim: 8, lower bound: -12.3203828, upper bound: 12.3203833
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 8.49
Output dim: 8, lower bound: -12.3203824, upper bound: 12.3203836

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

Time for backsubstitution: 1.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3200450, upper bound: 12.3200443
time: 3.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3200450, upper bound: 12.3200443
time: 3.10 seconds

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

Time for backsubstitution: 1.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3200449, upper bound: 12.3200446
time: 2.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3200449, upper bound: 12.3200445
time: 3.29 seconds

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

Time for backsubstitution: 1.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3200453, upper bound: 12.3200443
time: 3.36 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -12.3200453, upper bound: 12.3200441
time: 3.96 seconds

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

Time for backsubstitution: 1.67 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 6.87 + 593.61 = 600.47 seconds
