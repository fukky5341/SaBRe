## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 12)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.15154795000000001


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0402414, 0.0519706, -0.0402414, 0.0519706, -0.0922120, 0.0922120)
1: (0.8656265, 1.0395867, 0.8656265, 1.0395867, -0.1739601, 0.1739601)
2: (-0.0293665, 0.0697505, -0.0293665, 0.0697505, -0.0991171, 0.0991171)
3: (-0.0263494, 0.0377974, -0.0263494, 0.0377974, -0.0641469, 0.0641469)
4: (-0.0432664, 0.0205233, -0.0432664, 0.0205233, -0.0637898, 0.0637898)
5: (-0.0237454, 0.0481305, -0.0237454, 0.0481305, -0.0718759, 0.0718759)
6: (-0.0584624, 0.0301437, -0.0584624, 0.0301437, -0.0886062, 0.0886062)
7: (-0.0423451, 0.0703274, -0.0423451, 0.0703274, -0.1126725, 0.1126725)
8: (-0.0237534, 0.0443120, -0.0237534, 0.0443120, -0.0680653, 0.0680653)
9: (-0.0520953, 0.0472220, -0.0520953, 0.0472220, -0.0993173, 0.0993173)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.13 + 2.14 = 3.27 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.1562350, upper bound: 0.1562350

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 208

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 127

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1542610, upper bound: 0.1542610
time: 1.04 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1542610, upper bound: 0.1542610
time: 1.03 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 2.09 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 2.09
Output dim: 1, lower bound: -0.1542610, upper bound: 0.1542610
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 2.09
Output dim: 1, lower bound: -0.1542610, upper bound: 0.1542610

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.0402414, 0.0519706, -0.0402414, 0.0519706, -0.0922120, 0.0922120
1: 0.8656265, 1.0395867, 0.8656265, 1.0395867, -0.1739601, 0.1739601
2: -0.0293665, 0.0697505, -0.0293665, 0.0697505, -0.0991171, 0.0991171
3: -0.0263494, 0.0377974, -0.0263494, 0.0377974, -0.0641469, 0.0641469
4: -0.0432664, 0.0205233, -0.0432664, 0.0205233, -0.0637898, 0.0637898
5: -0.0237454, 0.0481305, -0.0237454, 0.0481305, -0.0718759, 0.0718759
6: -0.0584624, 0.0301437, -0.0584624, 0.0301437, -0.0886062, 0.0886062
7: -0.0423451, 0.0703274, -0.0423451, 0.0703274, -0.1126725, 0.1126725
8: -0.0237534, 0.0443120, -0.0237534, 0.0443120, -0.0680653, 0.0680653
9: -0.0520953, 0.0472220, -0.0520953, 0.0472220, -0.0993173, 0.0993173

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 68

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.1493903, upper bound: 0.1493903
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.1493903, upper bound: 0.1493903
time: 0.94 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.0402414, 0.0519706, -0.0402414, 0.0519706, -0.0922120, 0.0922120
1: 0.8656265, 1.0395867, 0.8656265, 1.0395867, -0.1739601, 0.1739601
2: -0.0293665, 0.0697505, -0.0293665, 0.0697505, -0.0991171, 0.0991171
3: -0.0263494, 0.0377974, -0.0263494, 0.0377974, -0.0641469, 0.0641469
4: -0.0432664, 0.0205233, -0.0432664, 0.0205233, -0.0637898, 0.0637898
5: -0.0237454, 0.0481305, -0.0237454, 0.0481305, -0.0718759, 0.0718759
6: -0.0584624, 0.0301437, -0.0584624, 0.0301437, -0.0886062, 0.0886062
7: -0.0423451, 0.0703274, -0.0423451, 0.0703274, -0.1126725, 0.1126725
8: -0.0237534, 0.0443120, -0.0237534, 0.0443120, -0.0680653, 0.0680653
9: -0.0520953, 0.0472220, -0.0520953, 0.0472220, -0.0993173, 0.0993173

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 169

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.1503748, upper bound: 0.1503748
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.1503748, upper bound: 0.1503748
time: 1.02 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 3.11 seconds
DS_DSZ1_DSZ1, status: Status.VERIFIED, split count: 2, time: 3.11
Output dim: 1, lower bound: -0.1493903, upper bound: 0.1493903
DS_DSZ1_DSZ2, status: Status.VERIFIED, split count: 2, time: 3.11
Output dim: 1, lower bound: -0.1493903, upper bound: 0.1493903
DS_DSZ2_DSZ1, status: Status.VERIFIED, split count: 2, time: 3.11
Output dim: 1, lower bound: -0.1503748, upper bound: 0.1503748
DS_DSZ2_DSZ2, status: Status.VERIFIED, split count: 2, time: 3.11
Output dim: 1, lower bound: -0.1503748, upper bound: 0.1503748

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 3.27 + 8.10 = 11.37 seconds
