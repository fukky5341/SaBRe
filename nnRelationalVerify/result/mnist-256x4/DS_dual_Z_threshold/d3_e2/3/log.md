## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 9.007671511800002


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-7.6137094, 5.8590560, -7.6137094, 5.8590560, -13.4727650, 13.4727640)
1: (-6.1292791, 5.3075094, -6.1292791, 5.3075094, -11.4367886, 11.4367886)
2: (-7.4691033, 4.5115442, -7.4691033, 4.5115442, -11.9806461, 11.9806480)
3: (-9.0654783, 4.4164257, -9.0654783, 4.4164257, -13.4819021, 13.4819012)
4: (-8.2032833, 6.3942995, -8.2032833, 6.3942995, -14.5975828, 14.5975819)
5: (-6.5569053, 5.5308518, -6.5569053, 5.5308518, -12.0877571, 12.0877571)
6: (-6.6569500, 7.0587082, -6.6569500, 7.0587082, -13.7156572, 13.7156572)
7: (-8.3428059, 4.1933599, -8.3428059, 4.1933599, -12.5361652, 12.5361652)
8: (-8.1921329, 6.0312138, -8.1921329, 6.0312138, -14.2233467, 14.2233467)
9: (-6.3373275, 6.7819953, -6.3373275, 6.7819953, -13.1193199, 13.1193218)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.38 + 5.05 = 7.43 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -9.0166882, upper bound: 9.0166883

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0095889, upper bound: 9.0095889
time: 3.23 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0095889, upper bound: 9.0095889
time: 3.23 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 6.73 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 6.73
Output dim: 7, lower bound: -9.0095889, upper bound: 9.0095889
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 6.73
Output dim: 7, lower bound: -9.0095889, upper bound: 9.0095889

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -7.6137094, 5.8590560, -7.6137094, 5.8590560, -13.4727650, 13.4727640
1: -6.1292791, 5.3075094, -6.1292791, 5.3075094, -11.4367886, 11.4367886
2: -7.4691033, 4.5115442, -7.4691033, 4.5115442, -11.9806461, 11.9806480
3: -9.0654783, 4.4164257, -9.0654783, 4.4164257, -13.4819021, 13.4819012
4: -8.2032833, 6.3942995, -8.2032833, 6.3942995, -14.5975828, 14.5975819
5: -6.5569053, 5.5308518, -6.5569053, 5.5308518, -12.0877571, 12.0877571
6: -6.6569500, 7.0587082, -6.6569500, 7.0587082, -13.7156572, 13.7156572
7: -8.3428059, 4.1933599, -8.3428059, 4.1933599, -12.5361652, 12.5361652
8: -8.1921329, 6.0312138, -8.1921329, 6.0312138, -14.2233467, 14.2233467
9: -6.3373275, 6.7819953, -6.3373275, 6.7819953, -13.1193199, 13.1193218

Time for backsubstitution: 2.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 138

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0094059, upper bound: 9.0094059
time: 1.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0094060, upper bound: 9.0094060
time: 2.08 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -7.6137094, 5.8590560, -7.6137094, 5.8590560, -13.4727650, 13.4727640
1: -6.1292791, 5.3075094, -6.1292791, 5.3075094, -11.4367886, 11.4367886
2: -7.4691033, 4.5115442, -7.4691033, 4.5115442, -11.9806461, 11.9806480
3: -9.0654783, 4.4164257, -9.0654783, 4.4164257, -13.4819021, 13.4819012
4: -8.2032833, 6.3942995, -8.2032833, 6.3942995, -14.5975828, 14.5975819
5: -6.5569053, 5.5308518, -6.5569053, 5.5308518, -12.0877571, 12.0877571
6: -6.6569500, 7.0587082, -6.6569500, 7.0587082, -13.7156572, 13.7156572
7: -8.3428059, 4.1933599, -8.3428059, 4.1933599, -12.5361652, 12.5361652
8: -8.1921329, 6.0312138, -8.1921329, 6.0312138, -14.2233467, 14.2233467
9: -6.3373275, 6.7819953, -6.3373275, 6.7819953, -13.1193199, 13.1193218

Time for backsubstitution: 2.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 138

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0094059, upper bound: 9.0094059
time: 1.90 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0094060, upper bound: 9.0094060
time: 1.99 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 8.42 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 8.42
Output dim: 7, lower bound: -9.0094059, upper bound: 9.0094059
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 8.42
Output dim: 7, lower bound: -9.0094060, upper bound: 9.0094060
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 8.42
Output dim: 7, lower bound: -9.0094059, upper bound: 9.0094059
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 8.42
Output dim: 7, lower bound: -9.0094060, upper bound: 9.0094060

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6137094, 5.8590560, -7.6137094, 5.8590560, -13.4727650, 13.4727640
1: -6.1292791, 5.3075094, -6.1292791, 5.3075094, -11.4367886, 11.4367886
2: -7.4691033, 4.5115442, -7.4691033, 4.5115442, -11.9806461, 11.9806480
3: -9.0654783, 4.4164257, -9.0654783, 4.4164257, -13.4819021, 13.4819012
4: -8.2032833, 6.3942995, -8.2032833, 6.3942995, -14.5975828, 14.5975819
5: -6.5569053, 5.5308518, -6.5569053, 5.5308518, -12.0877571, 12.0877571
6: -6.6569500, 7.0587082, -6.6569500, 7.0587082, -13.7156572, 13.7156572
7: -8.3428059, 4.1933599, -8.3428059, 4.1933599, -12.5361652, 12.5361652
8: -8.1921329, 6.0312138, -8.1921329, 6.0312138, -14.2233467, 14.2233467
9: -6.3373275, 6.7819953, -6.3373275, 6.7819953, -13.1193199, 13.1193218

Time for backsubstitution: 1.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Candidate
type: DSZ, layer: 1, pos: 153

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093946, upper bound: 9.0093943
time: 2.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093943, upper bound: 9.0093946
time: 2.74 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6137094, 5.8590560, -7.6137094, 5.8590560, -13.4727650, 13.4727640
1: -6.1292791, 5.3075094, -6.1292791, 5.3075094, -11.4367886, 11.4367886
2: -7.4691033, 4.5115442, -7.4691033, 4.5115442, -11.9806461, 11.9806480
3: -9.0654783, 4.4164257, -9.0654783, 4.4164257, -13.4819021, 13.4819012
4: -8.2032833, 6.3942995, -8.2032833, 6.3942995, -14.5975828, 14.5975819
5: -6.5569053, 5.5308518, -6.5569053, 5.5308518, -12.0877571, 12.0877571
6: -6.6569500, 7.0587082, -6.6569500, 7.0587082, -13.7156572, 13.7156572
7: -8.3428059, 4.1933599, -8.3428059, 4.1933599, -12.5361652, 12.5361652
8: -8.1921329, 6.0312138, -8.1921329, 6.0312138, -14.2233467, 14.2233467
9: -6.3373275, 6.7819953, -6.3373275, 6.7819953, -13.1193199, 13.1193218

Time for backsubstitution: 1.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Candidate
type: DSZ, layer: 1, pos: 153

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093946, upper bound: 9.0093944
time: 2.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093943, upper bound: 9.0093946
time: 2.00 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6137094, 5.8590560, -7.6137094, 5.8590560, -13.4727650, 13.4727640
1: -6.1292791, 5.3075094, -6.1292791, 5.3075094, -11.4367886, 11.4367886
2: -7.4691033, 4.5115442, -7.4691033, 4.5115442, -11.9806461, 11.9806480
3: -9.0654783, 4.4164257, -9.0654783, 4.4164257, -13.4819021, 13.4819012
4: -8.2032833, 6.3942995, -8.2032833, 6.3942995, -14.5975828, 14.5975819
5: -6.5569053, 5.5308518, -6.5569053, 5.5308518, -12.0877571, 12.0877571
6: -6.6569500, 7.0587082, -6.6569500, 7.0587082, -13.7156572, 13.7156572
7: -8.3428059, 4.1933599, -8.3428059, 4.1933599, -12.5361652, 12.5361652
8: -8.1921329, 6.0312138, -8.1921329, 6.0312138, -14.2233467, 14.2233467
9: -6.3373275, 6.7819953, -6.3373275, 6.7819953, -13.1193199, 13.1193218

Time for backsubstitution: 2.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Candidate
type: DSZ, layer: 1, pos: 153

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093946, upper bound: 9.0093943
time: 2.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093943, upper bound: 9.0093946
time: 2.87 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6137094, 5.8590560, -7.6137094, 5.8590560, -13.4727650, 13.4727640
1: -6.1292791, 5.3075094, -6.1292791, 5.3075094, -11.4367886, 11.4367886
2: -7.4691033, 4.5115442, -7.4691033, 4.5115442, -11.9806461, 11.9806480
3: -9.0654783, 4.4164257, -9.0654783, 4.4164257, -13.4819021, 13.4819012
4: -8.2032833, 6.3942995, -8.2032833, 6.3942995, -14.5975828, 14.5975819
5: -6.5569053, 5.5308518, -6.5569053, 5.5308518, -12.0877571, 12.0877571
6: -6.6569500, 7.0587082, -6.6569500, 7.0587082, -13.7156572, 13.7156572
7: -8.3428059, 4.1933599, -8.3428059, 4.1933599, -12.5361652, 12.5361652
8: -8.1921329, 6.0312138, -8.1921329, 6.0312138, -14.2233467, 14.2233467
9: -6.3373275, 6.7819953, -6.3373275, 6.7819953, -13.1193199, 13.1193218

Time for backsubstitution: 1.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Candidate
type: DSZ, layer: 1, pos: 153

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093946, upper bound: 9.0093944
time: 2.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093943, upper bound: 9.0093946
time: 1.95 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 6.89 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 6.89
Output dim: 7, lower bound: -9.0093946, upper bound: 9.0093943
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 6.89
Output dim: 7, lower bound: -9.0093943, upper bound: 9.0093946
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 6.89
Output dim: 7, lower bound: -9.0093946, upper bound: 9.0093944
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 6.89
Output dim: 7, lower bound: -9.0093943, upper bound: 9.0093946
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 6.89
Output dim: 7, lower bound: -9.0093946, upper bound: 9.0093943
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 6.89
Output dim: 7, lower bound: -9.0093943, upper bound: 9.0093946
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 6.89
Output dim: 7, lower bound: -9.0093946, upper bound: 9.0093944
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 6.89
Output dim: 7, lower bound: -9.0093943, upper bound: 9.0093946

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6137094, 5.8590560, -7.6137094, 5.8590560, -13.4727650, 13.4727640
1: -6.1292791, 5.3075094, -6.1292791, 5.3075094, -11.4367886, 11.4367886
2: -7.4691033, 4.5115442, -7.4691033, 4.5115442, -11.9806461, 11.9806480
3: -9.0654783, 4.4164257, -9.0654783, 4.4164257, -13.4819021, 13.4819012
4: -8.2032833, 6.3942995, -8.2032833, 6.3942995, -14.5975828, 14.5975819
5: -6.5569053, 5.5308518, -6.5569053, 5.5308518, -12.0877571, 12.0877571
6: -6.6569500, 7.0587082, -6.6569500, 7.0587082, -13.7156572, 13.7156572
7: -8.3428059, 4.1933599, -8.3428059, 4.1933599, -12.5361652, 12.5361652
8: -8.1921329, 6.0312138, -8.1921329, 6.0312138, -14.2233467, 14.2233467
9: -6.3373275, 6.7819953, -6.3373275, 6.7819953, -13.1193199, 13.1193218

Time for backsubstitution: 1.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093946, upper bound: 9.0093925
time: 1.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093927, upper bound: 9.0093944
time: 2.46 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6137094, 5.8590560, -7.6137094, 5.8590560, -13.4727650, 13.4727640
1: -6.1292791, 5.3075094, -6.1292791, 5.3075094, -11.4367886, 11.4367886
2: -7.4691033, 4.5115442, -7.4691033, 4.5115442, -11.9806461, 11.9806480
3: -9.0654783, 4.4164257, -9.0654783, 4.4164257, -13.4819021, 13.4819012
4: -8.2032833, 6.3942995, -8.2032833, 6.3942995, -14.5975828, 14.5975819
5: -6.5569053, 5.5308518, -6.5569053, 5.5308518, -12.0877571, 12.0877571
6: -6.6569500, 7.0587082, -6.6569500, 7.0587082, -13.7156572, 13.7156572
7: -8.3428059, 4.1933599, -8.3428059, 4.1933599, -12.5361652, 12.5361652
8: -8.1921329, 6.0312138, -8.1921329, 6.0312138, -14.2233467, 14.2233467
9: -6.3373275, 6.7819953, -6.3373275, 6.7819953, -13.1193199, 13.1193218

Time for backsubstitution: 2.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093944, upper bound: 9.0093927
time: 2.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093925, upper bound: 9.0093946
time: 8.32 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6137094, 5.8590560, -7.6137094, 5.8590560, -13.4727650, 13.4727640
1: -6.1292791, 5.3075094, -6.1292791, 5.3075094, -11.4367886, 11.4367886
2: -7.4691033, 4.5115442, -7.4691033, 4.5115442, -11.9806461, 11.9806480
3: -9.0654783, 4.4164257, -9.0654783, 4.4164257, -13.4819021, 13.4819012
4: -8.2032833, 6.3942995, -8.2032833, 6.3942995, -14.5975828, 14.5975819
5: -6.5569053, 5.5308518, -6.5569053, 5.5308518, -12.0877571, 12.0877571
6: -6.6569500, 7.0587082, -6.6569500, 7.0587082, -13.7156572, 13.7156572
7: -8.3428059, 4.1933599, -8.3428059, 4.1933599, -12.5361652, 12.5361652
8: -8.1921329, 6.0312138, -8.1921329, 6.0312138, -14.2233467, 14.2233467
9: -6.3373275, 6.7819953, -6.3373275, 6.7819953, -13.1193199, 13.1193218

Time for backsubstitution: 2.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093946, upper bound: 9.0093925
time: 2.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093927, upper bound: 9.0093943
time: 2.42 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6137094, 5.8590560, -7.6137094, 5.8590560, -13.4727650, 13.4727640
1: -6.1292791, 5.3075094, -6.1292791, 5.3075094, -11.4367886, 11.4367886
2: -7.4691033, 4.5115442, -7.4691033, 4.5115442, -11.9806461, 11.9806480
3: -9.0654783, 4.4164257, -9.0654783, 4.4164257, -13.4819021, 13.4819012
4: -8.2032833, 6.3942995, -8.2032833, 6.3942995, -14.5975828, 14.5975819
5: -6.5569053, 5.5308518, -6.5569053, 5.5308518, -12.0877571, 12.0877571
6: -6.6569500, 7.0587082, -6.6569500, 7.0587082, -13.7156572, 13.7156572
7: -8.3428059, 4.1933599, -8.3428059, 4.1933599, -12.5361652, 12.5361652
8: -8.1921329, 6.0312138, -8.1921329, 6.0312138, -14.2233467, 14.2233467
9: -6.3373275, 6.7819953, -6.3373275, 6.7819953, -13.1193199, 13.1193218

Time for backsubstitution: 2.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093944, upper bound: 9.0093927
time: 2.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093925, upper bound: 9.0093946
time: 1.71 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6137094, 5.8590560, -7.6137094, 5.8590560, -13.4727650, 13.4727640
1: -6.1292791, 5.3075094, -6.1292791, 5.3075094, -11.4367886, 11.4367886
2: -7.4691033, 4.5115442, -7.4691033, 4.5115442, -11.9806461, 11.9806480
3: -9.0654783, 4.4164257, -9.0654783, 4.4164257, -13.4819021, 13.4819012
4: -8.2032833, 6.3942995, -8.2032833, 6.3942995, -14.5975828, 14.5975819
5: -6.5569053, 5.5308518, -6.5569053, 5.5308518, -12.0877571, 12.0877571
6: -6.6569500, 7.0587082, -6.6569500, 7.0587082, -13.7156572, 13.7156572
7: -8.3428059, 4.1933599, -8.3428059, 4.1933599, -12.5361652, 12.5361652
8: -8.1921329, 6.0312138, -8.1921329, 6.0312138, -14.2233467, 14.2233467
9: -6.3373275, 6.7819953, -6.3373275, 6.7819953, -13.1193199, 13.1193218

Time for backsubstitution: 1.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093946, upper bound: 9.0093925
time: 1.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093927, upper bound: 9.0093944
time: 2.30 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6137094, 5.8590560, -7.6137094, 5.8590560, -13.4727650, 13.4727640
1: -6.1292791, 5.3075094, -6.1292791, 5.3075094, -11.4367886, 11.4367886
2: -7.4691033, 4.5115442, -7.4691033, 4.5115442, -11.9806461, 11.9806480
3: -9.0654783, 4.4164257, -9.0654783, 4.4164257, -13.4819021, 13.4819012
4: -8.2032833, 6.3942995, -8.2032833, 6.3942995, -14.5975828, 14.5975819
5: -6.5569053, 5.5308518, -6.5569053, 5.5308518, -12.0877571, 12.0877571
6: -6.6569500, 7.0587082, -6.6569500, 7.0587082, -13.7156572, 13.7156572
7: -8.3428059, 4.1933599, -8.3428059, 4.1933599, -12.5361652, 12.5361652
8: -8.1921329, 6.0312138, -8.1921329, 6.0312138, -14.2233467, 14.2233467
9: -6.3373275, 6.7819953, -6.3373275, 6.7819953, -13.1193199, 13.1193218

Time for backsubstitution: 1.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093944, upper bound: 9.0093927
time: 2.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093925, upper bound: 9.0093946
time: 8.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6137094, 5.8590560, -7.6137094, 5.8590560, -13.4727650, 13.4727640
1: -6.1292791, 5.3075094, -6.1292791, 5.3075094, -11.4367886, 11.4367886
2: -7.4691033, 4.5115442, -7.4691033, 4.5115442, -11.9806461, 11.9806480
3: -9.0654783, 4.4164257, -9.0654783, 4.4164257, -13.4819021, 13.4819012
4: -8.2032833, 6.3942995, -8.2032833, 6.3942995, -14.5975828, 14.5975819
5: -6.5569053, 5.5308518, -6.5569053, 5.5308518, -12.0877571, 12.0877571
6: -6.6569500, 7.0587082, -6.6569500, 7.0587082, -13.7156572, 13.7156572
7: -8.3428059, 4.1933599, -8.3428059, 4.1933599, -12.5361652, 12.5361652
8: -8.1921329, 6.0312138, -8.1921329, 6.0312138, -14.2233467, 14.2233467
9: -6.3373275, 6.7819953, -6.3373275, 6.7819953, -13.1193199, 13.1193218

Time for backsubstitution: 1.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093946, upper bound: 9.0093925
time: 1.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093927, upper bound: 9.0093943
time: 2.23 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6137094, 5.8590560, -7.6137094, 5.8590560, -13.4727650, 13.4727640
1: -6.1292791, 5.3075094, -6.1292791, 5.3075094, -11.4367886, 11.4367886
2: -7.4691033, 4.5115442, -7.4691033, 4.5115442, -11.9806461, 11.9806480
3: -9.0654783, 4.4164257, -9.0654783, 4.4164257, -13.4819021, 13.4819012
4: -8.2032833, 6.3942995, -8.2032833, 6.3942995, -14.5975828, 14.5975819
5: -6.5569053, 5.5308518, -6.5569053, 5.5308518, -12.0877571, 12.0877571
6: -6.6569500, 7.0587082, -6.6569500, 7.0587082, -13.7156572, 13.7156572
7: -8.3428059, 4.1933599, -8.3428059, 4.1933599, -12.5361652, 12.5361652
8: -8.1921329, 6.0312138, -8.1921329, 6.0312138, -14.2233467, 14.2233467
9: -6.3373275, 6.7819953, -6.3373275, 6.7819953, -13.1193199, 13.1193218

Time for backsubstitution: 1.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093944, upper bound: 9.0093927
time: 2.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093925, upper bound: 9.0093946
time: 1.79 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 6.62 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 6.62
Output dim: 7, lower bound: -9.0093946, upper bound: 9.0093925
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 6.62
Output dim: 7, lower bound: -9.0093927, upper bound: 9.0093944
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 6.62
Output dim: 7, lower bound: -9.0093944, upper bound: 9.0093927
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 6.62
Output dim: 7, lower bound: -9.0093925, upper bound: 9.0093946
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 6.62
Output dim: 7, lower bound: -9.0093946, upper bound: 9.0093925
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 6.62
Output dim: 7, lower bound: -9.0093927, upper bound: 9.0093943
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 6.62
Output dim: 7, lower bound: -9.0093944, upper bound: 9.0093927
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 6.62
Output dim: 7, lower bound: -9.0093925, upper bound: 9.0093946
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 6.62
Output dim: 7, lower bound: -9.0093946, upper bound: 9.0093925
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 6.62
Output dim: 7, lower bound: -9.0093927, upper bound: 9.0093944
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 6.62
Output dim: 7, lower bound: -9.0093944, upper bound: 9.0093927
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 6.62
Output dim: 7, lower bound: -9.0093925, upper bound: 9.0093946
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 6.62
Output dim: 7, lower bound: -9.0093946, upper bound: 9.0093925
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 6.62
Output dim: 7, lower bound: -9.0093927, upper bound: 9.0093943
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 6.62
Output dim: 7, lower bound: -9.0093944, upper bound: 9.0093927
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 6.62
Output dim: 7, lower bound: -9.0093925, upper bound: 9.0093946

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6137094, 5.8590560, -7.6137094, 5.8590560, -13.4727650, 13.4727640
1: -6.1292791, 5.3075094, -6.1292791, 5.3075094, -11.4367886, 11.4367886
2: -7.4691033, 4.5115442, -7.4691033, 4.5115442, -11.9806461, 11.9806480
3: -9.0654783, 4.4164257, -9.0654783, 4.4164257, -13.4819021, 13.4819012
4: -8.2032833, 6.3942995, -8.2032833, 6.3942995, -14.5975828, 14.5975819
5: -6.5569053, 5.5308518, -6.5569053, 5.5308518, -12.0877571, 12.0877571
6: -6.6569500, 7.0587082, -6.6569500, 7.0587082, -13.7156572, 13.7156572
7: -8.3428059, 4.1933599, -8.3428059, 4.1933599, -12.5361652, 12.5361652
8: -8.1921329, 6.0312138, -8.1921329, 6.0312138, -14.2233467, 14.2233467
9: -6.3373275, 6.7819953, -6.3373275, 6.7819953, -13.1193199, 13.1193218

Time for backsubstitution: 2.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.27 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Candidate
type: DSZ, layer: 1, pos: 68

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093946, upper bound: 9.0093925
time: 2.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093946, upper bound: 9.0093925
time: 1.95 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6137094, 5.8590560, -7.6137094, 5.8590560, -13.4727650, 13.4727640
1: -6.1292791, 5.3075094, -6.1292791, 5.3075094, -11.4367886, 11.4367886
2: -7.4691033, 4.5115442, -7.4691033, 4.5115442, -11.9806461, 11.9806480
3: -9.0654783, 4.4164257, -9.0654783, 4.4164257, -13.4819021, 13.4819012
4: -8.2032833, 6.3942995, -8.2032833, 6.3942995, -14.5975828, 14.5975819
5: -6.5569053, 5.5308518, -6.5569053, 5.5308518, -12.0877571, 12.0877571
6: -6.6569500, 7.0587082, -6.6569500, 7.0587082, -13.7156572, 13.7156572
7: -8.3428059, 4.1933599, -8.3428059, 4.1933599, -12.5361652, 12.5361652
8: -8.1921329, 6.0312138, -8.1921329, 6.0312138, -14.2233467, 14.2233467
9: -6.3373275, 6.7819953, -6.3373275, 6.7819953, -13.1193199, 13.1193218

Time for backsubstitution: 1.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Candidate
type: DSZ, layer: 1, pos: 68

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093926, upper bound: 9.0093944
time: 2.13 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093927, upper bound: 9.0093943
time: 2.39 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6137094, 5.8590560, -7.6137094, 5.8590560, -13.4727650, 13.4727640
1: -6.1292791, 5.3075094, -6.1292791, 5.3075094, -11.4367886, 11.4367886
2: -7.4691033, 4.5115442, -7.4691033, 4.5115442, -11.9806461, 11.9806480
3: -9.0654783, 4.4164257, -9.0654783, 4.4164257, -13.4819021, 13.4819012
4: -8.2032833, 6.3942995, -8.2032833, 6.3942995, -14.5975828, 14.5975819
5: -6.5569053, 5.5308518, -6.5569053, 5.5308518, -12.0877571, 12.0877571
6: -6.6569500, 7.0587082, -6.6569500, 7.0587082, -13.7156572, 13.7156572
7: -8.3428059, 4.1933599, -8.3428059, 4.1933599, -12.5361652, 12.5361652
8: -8.1921329, 6.0312138, -8.1921329, 6.0312138, -14.2233467, 14.2233467
9: -6.3373275, 6.7819953, -6.3373275, 6.7819953, -13.1193199, 13.1193218

Time for backsubstitution: 1.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Candidate
type: DSZ, layer: 1, pos: 68

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093943, upper bound: 9.0093927
time: 1.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093943, upper bound: 9.0093927
time: 2.56 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6137094, 5.8590560, -7.6137094, 5.8590560, -13.4727650, 13.4727640
1: -6.1292791, 5.3075094, -6.1292791, 5.3075094, -11.4367886, 11.4367886
2: -7.4691033, 4.5115442, -7.4691033, 4.5115442, -11.9806461, 11.9806480
3: -9.0654783, 4.4164257, -9.0654783, 4.4164257, -13.4819021, 13.4819012
4: -8.2032833, 6.3942995, -8.2032833, 6.3942995, -14.5975828, 14.5975819
5: -6.5569053, 5.5308518, -6.5569053, 5.5308518, -12.0877571, 12.0877571
6: -6.6569500, 7.0587082, -6.6569500, 7.0587082, -13.7156572, 13.7156572
7: -8.3428059, 4.1933599, -8.3428059, 4.1933599, -12.5361652, 12.5361652
8: -8.1921329, 6.0312138, -8.1921329, 6.0312138, -14.2233467, 14.2233467
9: -6.3373275, 6.7819953, -6.3373275, 6.7819953, -13.1193199, 13.1193218

Time for backsubstitution: 1.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Candidate
type: DSZ, layer: 1, pos: 68

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093924, upper bound: 9.0093946
time: 1.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093925, upper bound: 9.0093946
time: 2.09 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6137094, 5.8590560, -7.6137094, 5.8590560, -13.4727650, 13.4727640
1: -6.1292791, 5.3075094, -6.1292791, 5.3075094, -11.4367886, 11.4367886
2: -7.4691033, 4.5115442, -7.4691033, 4.5115442, -11.9806461, 11.9806480
3: -9.0654783, 4.4164257, -9.0654783, 4.4164257, -13.4819021, 13.4819012
4: -8.2032833, 6.3942995, -8.2032833, 6.3942995, -14.5975828, 14.5975819
5: -6.5569053, 5.5308518, -6.5569053, 5.5308518, -12.0877571, 12.0877571
6: -6.6569500, 7.0587082, -6.6569500, 7.0587082, -13.7156572, 13.7156572
7: -8.3428059, 4.1933599, -8.3428059, 4.1933599, -12.5361652, 12.5361652
8: -8.1921329, 6.0312138, -8.1921329, 6.0312138, -14.2233467, 14.2233467
9: -6.3373275, 6.7819953, -6.3373275, 6.7819953, -13.1193199, 13.1193218

Time for backsubstitution: 1.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Candidate
type: DSZ, layer: 1, pos: 68

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093946, upper bound: 9.0093925
time: 3.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093946, upper bound: 9.0093924
time: 2.33 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6137094, 5.8590560, -7.6137094, 5.8590560, -13.4727650, 13.4727640
1: -6.1292791, 5.3075094, -6.1292791, 5.3075094, -11.4367886, 11.4367886
2: -7.4691033, 4.5115442, -7.4691033, 4.5115442, -11.9806461, 11.9806480
3: -9.0654783, 4.4164257, -9.0654783, 4.4164257, -13.4819021, 13.4819012
4: -8.2032833, 6.3942995, -8.2032833, 6.3942995, -14.5975828, 14.5975819
5: -6.5569053, 5.5308518, -6.5569053, 5.5308518, -12.0877571, 12.0877571
6: -6.6569500, 7.0587082, -6.6569500, 7.0587082, -13.7156572, 13.7156572
7: -8.3428059, 4.1933599, -8.3428059, 4.1933599, -12.5361652, 12.5361652
8: -8.1921329, 6.0312138, -8.1921329, 6.0312138, -14.2233467, 14.2233467
9: -6.3373275, 6.7819953, -6.3373275, 6.7819953, -13.1193199, 13.1193218

Time for backsubstitution: 1.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Candidate
type: DSZ, layer: 1, pos: 68

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093927, upper bound: 9.0093944
time: 2.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093927, upper bound: 9.0093943
time: 2.54 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6137094, 5.8590560, -7.6137094, 5.8590560, -13.4727650, 13.4727640
1: -6.1292791, 5.3075094, -6.1292791, 5.3075094, -11.4367886, 11.4367886
2: -7.4691033, 4.5115442, -7.4691033, 4.5115442, -11.9806461, 11.9806480
3: -9.0654783, 4.4164257, -9.0654783, 4.4164257, -13.4819021, 13.4819012
4: -8.2032833, 6.3942995, -8.2032833, 6.3942995, -14.5975828, 14.5975819
5: -6.5569053, 5.5308518, -6.5569053, 5.5308518, -12.0877571, 12.0877571
6: -6.6569500, 7.0587082, -6.6569500, 7.0587082, -13.7156572, 13.7156572
7: -8.3428059, 4.1933599, -8.3428059, 4.1933599, -12.5361652, 12.5361652
8: -8.1921329, 6.0312138, -8.1921329, 6.0312138, -14.2233467, 14.2233467
9: -6.3373275, 6.7819953, -6.3373275, 6.7819953, -13.1193199, 13.1193218

Time for backsubstitution: 1.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Candidate
type: DSZ, layer: 1, pos: 68

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093943, upper bound: 9.0093927
time: 1.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093943, upper bound: 9.0093926
time: 1.98 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6137094, 5.8590560, -7.6137094, 5.8590560, -13.4727650, 13.4727640
1: -6.1292791, 5.3075094, -6.1292791, 5.3075094, -11.4367886, 11.4367886
2: -7.4691033, 4.5115442, -7.4691033, 4.5115442, -11.9806461, 11.9806480
3: -9.0654783, 4.4164257, -9.0654783, 4.4164257, -13.4819021, 13.4819012
4: -8.2032833, 6.3942995, -8.2032833, 6.3942995, -14.5975828, 14.5975819
5: -6.5569053, 5.5308518, -6.5569053, 5.5308518, -12.0877571, 12.0877571
6: -6.6569500, 7.0587082, -6.6569500, 7.0587082, -13.7156572, 13.7156572
7: -8.3428059, 4.1933599, -8.3428059, 4.1933599, -12.5361652, 12.5361652
8: -8.1921329, 6.0312138, -8.1921329, 6.0312138, -14.2233467, 14.2233467
9: -6.3373275, 6.7819953, -6.3373275, 6.7819953, -13.1193199, 13.1193218

Time for backsubstitution: 1.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Candidate
type: DSZ, layer: 1, pos: 68

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093924, upper bound: 9.0093946
time: 1.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093925, upper bound: 9.0093946
time: 1.82 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6137094, 5.8590560, -7.6137094, 5.8590560, -13.4727650, 13.4727640
1: -6.1292791, 5.3075094, -6.1292791, 5.3075094, -11.4367886, 11.4367886
2: -7.4691033, 4.5115442, -7.4691033, 4.5115442, -11.9806461, 11.9806480
3: -9.0654783, 4.4164257, -9.0654783, 4.4164257, -13.4819021, 13.4819012
4: -8.2032833, 6.3942995, -8.2032833, 6.3942995, -14.5975828, 14.5975819
5: -6.5569053, 5.5308518, -6.5569053, 5.5308518, -12.0877571, 12.0877571
6: -6.6569500, 7.0587082, -6.6569500, 7.0587082, -13.7156572, 13.7156572
7: -8.3428059, 4.1933599, -8.3428059, 4.1933599, -12.5361652, 12.5361652
8: -8.1921329, 6.0312138, -8.1921329, 6.0312138, -14.2233467, 14.2233467
9: -6.3373275, 6.7819953, -6.3373275, 6.7819953, -13.1193199, 13.1193218

Time for backsubstitution: 1.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Candidate
type: DSZ, layer: 1, pos: 68

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093946, upper bound: 9.0093925
time: 2.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093946, upper bound: 9.0093925
time: 1.89 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6137094, 5.8590560, -7.6137094, 5.8590560, -13.4727650, 13.4727640
1: -6.1292791, 5.3075094, -6.1292791, 5.3075094, -11.4367886, 11.4367886
2: -7.4691033, 4.5115442, -7.4691033, 4.5115442, -11.9806461, 11.9806480
3: -9.0654783, 4.4164257, -9.0654783, 4.4164257, -13.4819021, 13.4819012
4: -8.2032833, 6.3942995, -8.2032833, 6.3942995, -14.5975828, 14.5975819
5: -6.5569053, 5.5308518, -6.5569053, 5.5308518, -12.0877571, 12.0877571
6: -6.6569500, 7.0587082, -6.6569500, 7.0587082, -13.7156572, 13.7156572
7: -8.3428059, 4.1933599, -8.3428059, 4.1933599, -12.5361652, 12.5361652
8: -8.1921329, 6.0312138, -8.1921329, 6.0312138, -14.2233467, 14.2233467
9: -6.3373275, 6.7819953, -6.3373275, 6.7819953, -13.1193199, 13.1193218

Time for backsubstitution: 1.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Candidate
type: DSZ, layer: 1, pos: 68

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093926, upper bound: 9.0093944
time: 2.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093927, upper bound: 9.0093943
time: 2.46 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6137094, 5.8590560, -7.6137094, 5.8590560, -13.4727650, 13.4727640
1: -6.1292791, 5.3075094, -6.1292791, 5.3075094, -11.4367886, 11.4367886
2: -7.4691033, 4.5115442, -7.4691033, 4.5115442, -11.9806461, 11.9806480
3: -9.0654783, 4.4164257, -9.0654783, 4.4164257, -13.4819021, 13.4819012
4: -8.2032833, 6.3942995, -8.2032833, 6.3942995, -14.5975828, 14.5975819
5: -6.5569053, 5.5308518, -6.5569053, 5.5308518, -12.0877571, 12.0877571
6: -6.6569500, 7.0587082, -6.6569500, 7.0587082, -13.7156572, 13.7156572
7: -8.3428059, 4.1933599, -8.3428059, 4.1933599, -12.5361652, 12.5361652
8: -8.1921329, 6.0312138, -8.1921329, 6.0312138, -14.2233467, 14.2233467
9: -6.3373275, 6.7819953, -6.3373275, 6.7819953, -13.1193199, 13.1193218

Time for backsubstitution: 1.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Candidate
type: DSZ, layer: 1, pos: 68

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093943, upper bound: 9.0093927
time: 1.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093943, upper bound: 9.0093927
time: 2.57 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6137094, 5.8590560, -7.6137094, 5.8590560, -13.4727650, 13.4727640
1: -6.1292791, 5.3075094, -6.1292791, 5.3075094, -11.4367886, 11.4367886
2: -7.4691033, 4.5115442, -7.4691033, 4.5115442, -11.9806461, 11.9806480
3: -9.0654783, 4.4164257, -9.0654783, 4.4164257, -13.4819021, 13.4819012
4: -8.2032833, 6.3942995, -8.2032833, 6.3942995, -14.5975828, 14.5975819
5: -6.5569053, 5.5308518, -6.5569053, 5.5308518, -12.0877571, 12.0877571
6: -6.6569500, 7.0587082, -6.6569500, 7.0587082, -13.7156572, 13.7156572
7: -8.3428059, 4.1933599, -8.3428059, 4.1933599, -12.5361652, 12.5361652
8: -8.1921329, 6.0312138, -8.1921329, 6.0312138, -14.2233467, 14.2233467
9: -6.3373275, 6.7819953, -6.3373275, 6.7819953, -13.1193199, 13.1193218

Time for backsubstitution: 1.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Candidate
type: DSZ, layer: 1, pos: 68

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093924, upper bound: 9.0093946
time: 1.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093925, upper bound: 9.0093946
time: 2.35 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6137094, 5.8590560, -7.6137094, 5.8590560, -13.4727650, 13.4727640
1: -6.1292791, 5.3075094, -6.1292791, 5.3075094, -11.4367886, 11.4367886
2: -7.4691033, 4.5115442, -7.4691033, 4.5115442, -11.9806461, 11.9806480
3: -9.0654783, 4.4164257, -9.0654783, 4.4164257, -13.4819021, 13.4819012
4: -8.2032833, 6.3942995, -8.2032833, 6.3942995, -14.5975828, 14.5975819
5: -6.5569053, 5.5308518, -6.5569053, 5.5308518, -12.0877571, 12.0877571
6: -6.6569500, 7.0587082, -6.6569500, 7.0587082, -13.7156572, 13.7156572
7: -8.3428059, 4.1933599, -8.3428059, 4.1933599, -12.5361652, 12.5361652
8: -8.1921329, 6.0312138, -8.1921329, 6.0312138, -14.2233467, 14.2233467
9: -6.3373275, 6.7819953, -6.3373275, 6.7819953, -13.1193199, 13.1193218

Time for backsubstitution: 2.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.27 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Candidate
type: DSZ, layer: 1, pos: 68

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093946, upper bound: 9.0093925
time: 3.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093946, upper bound: 9.0093924
time: 2.59 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6137094, 5.8590560, -7.6137094, 5.8590560, -13.4727650, 13.4727640
1: -6.1292791, 5.3075094, -6.1292791, 5.3075094, -11.4367886, 11.4367886
2: -7.4691033, 4.5115442, -7.4691033, 4.5115442, -11.9806461, 11.9806480
3: -9.0654783, 4.4164257, -9.0654783, 4.4164257, -13.4819021, 13.4819012
4: -8.2032833, 6.3942995, -8.2032833, 6.3942995, -14.5975828, 14.5975819
5: -6.5569053, 5.5308518, -6.5569053, 5.5308518, -12.0877571, 12.0877571
6: -6.6569500, 7.0587082, -6.6569500, 7.0587082, -13.7156572, 13.7156572
7: -8.3428059, 4.1933599, -8.3428059, 4.1933599, -12.5361652, 12.5361652
8: -8.1921329, 6.0312138, -8.1921329, 6.0312138, -14.2233467, 14.2233467
9: -6.3373275, 6.7819953, -6.3373275, 6.7819953, -13.1193199, 13.1193218

Time for backsubstitution: 2.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.28 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Candidate
type: DSZ, layer: 1, pos: 68

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093927, upper bound: 9.0093944
time: 2.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093927, upper bound: 9.0093943
time: 2.78 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6137094, 5.8590560, -7.6137094, 5.8590560, -13.4727650, 13.4727640
1: -6.1292791, 5.3075094, -6.1292791, 5.3075094, -11.4367886, 11.4367886
2: -7.4691033, 4.5115442, -7.4691033, 4.5115442, -11.9806461, 11.9806480
3: -9.0654783, 4.4164257, -9.0654783, 4.4164257, -13.4819021, 13.4819012
4: -8.2032833, 6.3942995, -8.2032833, 6.3942995, -14.5975828, 14.5975819
5: -6.5569053, 5.5308518, -6.5569053, 5.5308518, -12.0877571, 12.0877571
6: -6.6569500, 7.0587082, -6.6569500, 7.0587082, -13.7156572, 13.7156572
7: -8.3428059, 4.1933599, -8.3428059, 4.1933599, -12.5361652, 12.5361652
8: -8.1921329, 6.0312138, -8.1921329, 6.0312138, -14.2233467, 14.2233467
9: -6.3373275, 6.7819953, -6.3373275, 6.7819953, -13.1193199, 13.1193218

Time for backsubstitution: 2.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.27 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Candidate
type: DSZ, layer: 1, pos: 68

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093943, upper bound: 9.0093927
time: 1.92 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093943, upper bound: 9.0093926
time: 2.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6137094, 5.8590560, -7.6137094, 5.8590560, -13.4727650, 13.4727640
1: -6.1292791, 5.3075094, -6.1292791, 5.3075094, -11.4367886, 11.4367886
2: -7.4691033, 4.5115442, -7.4691033, 4.5115442, -11.9806461, 11.9806480
3: -9.0654783, 4.4164257, -9.0654783, 4.4164257, -13.4819021, 13.4819012
4: -8.2032833, 6.3942995, -8.2032833, 6.3942995, -14.5975828, 14.5975819
5: -6.5569053, 5.5308518, -6.5569053, 5.5308518, -12.0877571, 12.0877571
6: -6.6569500, 7.0587082, -6.6569500, 7.0587082, -13.7156572, 13.7156572
7: -8.3428059, 4.1933599, -8.3428059, 4.1933599, -12.5361652, 12.5361652
8: -8.1921329, 6.0312138, -8.1921329, 6.0312138, -14.2233467, 14.2233467
9: -6.3373275, 6.7819953, -6.3373275, 6.7819953, -13.1193199, 13.1193218

Time for backsubstitution: 2.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Candidate
type: DSZ, layer: 1, pos: 68

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093924, upper bound: 9.0093946
time: 1.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093925, upper bound: 9.0093946
time: 1.77 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 6.34 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 6.34
Output dim: 7, lower bound: -9.0093946, upper bound: 9.0093925
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 6.34
Output dim: 7, lower bound: -9.0093946, upper bound: 9.0093925
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 6.34
Output dim: 7, lower bound: -9.0093926, upper bound: 9.0093944
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 6.34
Output dim: 7, lower bound: -9.0093927, upper bound: 9.0093943
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 6.34
Output dim: 7, lower bound: -9.0093943, upper bound: 9.0093927
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 6.34
Output dim: 7, lower bound: -9.0093943, upper bound: 9.0093927
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 6.34
Output dim: 7, lower bound: -9.0093924, upper bound: 9.0093946
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 6.34
Output dim: 7, lower bound: -9.0093925, upper bound: 9.0093946
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 6.34
Output dim: 7, lower bound: -9.0093946, upper bound: 9.0093925
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 6.34
Output dim: 7, lower bound: -9.0093946, upper bound: 9.0093924
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 6.34
Output dim: 7, lower bound: -9.0093927, upper bound: 9.0093944
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 6.34
Output dim: 7, lower bound: -9.0093927, upper bound: 9.0093943
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 6.34
Output dim: 7, lower bound: -9.0093943, upper bound: 9.0093927
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 6.34
Output dim: 7, lower bound: -9.0093943, upper bound: 9.0093926
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 6.34
Output dim: 7, lower bound: -9.0093924, upper bound: 9.0093946
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 6.34
Output dim: 7, lower bound: -9.0093925, upper bound: 9.0093946
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 6.34
Output dim: 7, lower bound: -9.0093946, upper bound: 9.0093925
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 6.34
Output dim: 7, lower bound: -9.0093946, upper bound: 9.0093925
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 6.34
Output dim: 7, lower bound: -9.0093926, upper bound: 9.0093944
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 6.34
Output dim: 7, lower bound: -9.0093927, upper bound: 9.0093943
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 6.34
Output dim: 7, lower bound: -9.0093943, upper bound: 9.0093927
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 6.34
Output dim: 7, lower bound: -9.0093943, upper bound: 9.0093927
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 6.34
Output dim: 7, lower bound: -9.0093924, upper bound: 9.0093946
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 6.34
Output dim: 7, lower bound: -9.0093925, upper bound: 9.0093946
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 6.34
Output dim: 7, lower bound: -9.0093946, upper bound: 9.0093925
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 6.34
Output dim: 7, lower bound: -9.0093946, upper bound: 9.0093924
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 6.34
Output dim: 7, lower bound: -9.0093927, upper bound: 9.0093944
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 6.34
Output dim: 7, lower bound: -9.0093927, upper bound: 9.0093943
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 6.34
Output dim: 7, lower bound: -9.0093943, upper bound: 9.0093927
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 6.34
Output dim: 7, lower bound: -9.0093943, upper bound: 9.0093926
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 6.34
Output dim: 7, lower bound: -9.0093924, upper bound: 9.0093946
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 6.34
Output dim: 7, lower bound: -9.0093925, upper bound: 9.0093946

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6137094, 5.8590560, -7.6137094, 5.8590560, -13.4727650, 13.4727640
1: -6.1292791, 5.3075094, -6.1292791, 5.3075094, -11.4367886, 11.4367886
2: -7.4691033, 4.5115442, -7.4691033, 4.5115442, -11.9806461, 11.9806480
3: -9.0654783, 4.4164257, -9.0654783, 4.4164257, -13.4819021, 13.4819012
4: -8.2032833, 6.3942995, -8.2032833, 6.3942995, -14.5975828, 14.5975819
5: -6.5569053, 5.5308518, -6.5569053, 5.5308518, -12.0877571, 12.0877571
6: -6.6569500, 7.0587082, -6.6569500, 7.0587082, -13.7156572, 13.7156572
7: -8.3428059, 4.1933599, -8.3428059, 4.1933599, -12.5361652, 12.5361652
8: -8.1921329, 6.0312138, -8.1921329, 6.0312138, -14.2233467, 14.2233467
9: -6.3373275, 6.7819953, -6.3373275, 6.7819953, -13.1193199, 13.1193218

Time for backsubstitution: 1.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087835
time: 6.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087836
time: 2.92 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6137094, 5.8590560, -7.6137094, 5.8590560, -13.4727650, 13.4727640
1: -6.1292791, 5.3075094, -6.1292791, 5.3075094, -11.4367886, 11.4367886
2: -7.4691033, 4.5115442, -7.4691033, 4.5115442, -11.9806461, 11.9806480
3: -9.0654783, 4.4164257, -9.0654783, 4.4164257, -13.4819021, 13.4819012
4: -8.2032833, 6.3942995, -8.2032833, 6.3942995, -14.5975828, 14.5975819
5: -6.5569053, 5.5308518, -6.5569053, 5.5308518, -12.0877571, 12.0877571
6: -6.6569500, 7.0587082, -6.6569500, 7.0587082, -13.7156572, 13.7156572
7: -8.3428059, 4.1933599, -8.3428059, 4.1933599, -12.5361652, 12.5361652
8: -8.1921329, 6.0312138, -8.1921329, 6.0312138, -14.2233467, 14.2233467
9: -6.3373275, 6.7819953, -6.3373275, 6.7819953, -13.1193199, 13.1193218

Time for backsubstitution: 2.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087835
time: 2.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087835
time: 2.37 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6137094, 5.8590560, -7.6137094, 5.8590560, -13.4727650, 13.4727640
1: -6.1292791, 5.3075094, -6.1292791, 5.3075094, -11.4367886, 11.4367886
2: -7.4691033, 4.5115442, -7.4691033, 4.5115442, -11.9806461, 11.9806480
3: -9.0654783, 4.4164257, -9.0654783, 4.4164257, -13.4819021, 13.4819012
4: -8.2032833, 6.3942995, -8.2032833, 6.3942995, -14.5975828, 14.5975819
5: -6.5569053, 5.5308518, -6.5569053, 5.5308518, -12.0877571, 12.0877571
6: -6.6569500, 7.0587082, -6.6569500, 7.0587082, -13.7156572, 13.7156572
7: -8.3428059, 4.1933599, -8.3428059, 4.1933599, -12.5361652, 12.5361652
8: -8.1921329, 6.0312138, -8.1921329, 6.0312138, -14.2233467, 14.2233467
9: -6.3373275, 6.7819953, -6.3373275, 6.7819953, -13.1193199, 13.1193218

Time for backsubstitution: 1.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087841, upper bound: 9.0087844
time: 3.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087840, upper bound: 9.0087843
time: 2.15 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6137094, 5.8590560, -7.6137094, 5.8590560, -13.4727650, 13.4727640
1: -6.1292791, 5.3075094, -6.1292791, 5.3075094, -11.4367886, 11.4367886
2: -7.4691033, 4.5115442, -7.4691033, 4.5115442, -11.9806461, 11.9806480
3: -9.0654783, 4.4164257, -9.0654783, 4.4164257, -13.4819021, 13.4819012
4: -8.2032833, 6.3942995, -8.2032833, 6.3942995, -14.5975828, 14.5975819
5: -6.5569053, 5.5308518, -6.5569053, 5.5308518, -12.0877571, 12.0877571
6: -6.6569500, 7.0587082, -6.6569500, 7.0587082, -13.7156572, 13.7156572
7: -8.3428059, 4.1933599, -8.3428059, 4.1933599, -12.5361652, 12.5361652
8: -8.1921329, 6.0312138, -8.1921329, 6.0312138, -14.2233467, 14.2233467
9: -6.3373275, 6.7819953, -6.3373275, 6.7819953, -13.1193199, 13.1193218

Time for backsubstitution: 1.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087841, upper bound: 9.0087843
time: 1.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087841, upper bound: 9.0087843
time: 2.72 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6137094, 5.8590560, -7.6137094, 5.8590560, -13.4727650, 13.4727640
1: -6.1292791, 5.3075094, -6.1292791, 5.3075094, -11.4367886, 11.4367886
2: -7.4691033, 4.5115442, -7.4691033, 4.5115442, -11.9806461, 11.9806480
3: -9.0654783, 4.4164257, -9.0654783, 4.4164257, -13.4819021, 13.4819012
4: -8.2032833, 6.3942995, -8.2032833, 6.3942995, -14.5975828, 14.5975819
5: -6.5569053, 5.5308518, -6.5569053, 5.5308518, -12.0877571, 12.0877571
6: -6.6569500, 7.0587082, -6.6569500, 7.0587082, -13.7156572, 13.7156572
7: -8.3428059, 4.1933599, -8.3428059, 4.1933599, -12.5361652, 12.5361652
8: -8.1921329, 6.0312138, -8.1921329, 6.0312138, -14.2233467, 14.2233467
9: -6.3373275, 6.7819953, -6.3373275, 6.7819953, -13.1193199, 13.1193218

Time for backsubstitution: 1.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087843, upper bound: 9.0087841
time: 2.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087843, upper bound: 9.0087841
time: 1.70 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6137094, 5.8590560, -7.6137094, 5.8590560, -13.4727650, 13.4727640
1: -6.1292791, 5.3075094, -6.1292791, 5.3075094, -11.4367886, 11.4367886
2: -7.4691033, 4.5115442, -7.4691033, 4.5115442, -11.9806461, 11.9806480
3: -9.0654783, 4.4164257, -9.0654783, 4.4164257, -13.4819021, 13.4819012
4: -8.2032833, 6.3942995, -8.2032833, 6.3942995, -14.5975828, 14.5975819
5: -6.5569053, 5.5308518, -6.5569053, 5.5308518, -12.0877571, 12.0877571
6: -6.6569500, 7.0587082, -6.6569500, 7.0587082, -13.7156572, 13.7156572
7: -8.3428059, 4.1933599, -8.3428059, 4.1933599, -12.5361652, 12.5361652
8: -8.1921329, 6.0312138, -8.1921329, 6.0312138, -14.2233467, 14.2233467
9: -6.3373275, 6.7819953, -6.3373275, 6.7819953, -13.1193199, 13.1193218

Time for backsubstitution: 1.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087843, upper bound: 9.0087841
time: 1.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087843, upper bound: 9.0087841
time: 2.34 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6137094, 5.8590560, -7.6137094, 5.8590560, -13.4727650, 13.4727640
1: -6.1292791, 5.3075094, -6.1292791, 5.3075094, -11.4367886, 11.4367886
2: -7.4691033, 4.5115442, -7.4691033, 4.5115442, -11.9806461, 11.9806480
3: -9.0654783, 4.4164257, -9.0654783, 4.4164257, -13.4819021, 13.4819012
4: -8.2032833, 6.3942995, -8.2032833, 6.3942995, -14.5975828, 14.5975819
5: -6.5569053, 5.5308518, -6.5569053, 5.5308518, -12.0877571, 12.0877571
6: -6.6569500, 7.0587082, -6.6569500, 7.0587082, -13.7156572, 13.7156572
7: -8.3428059, 4.1933599, -8.3428059, 4.1933599, -12.5361652, 12.5361652
8: -8.1921329, 6.0312138, -8.1921329, 6.0312138, -14.2233467, 14.2233467
9: -6.3373275, 6.7819953, -6.3373275, 6.7819953, -13.1193199, 13.1193218

Time for backsubstitution: 1.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087835, upper bound: 9.0087846
time: 2.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087835, upper bound: 9.0087846
time: 1.63 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6137094, 5.8590560, -7.6137094, 5.8590560, -13.4727650, 13.4727640
1: -6.1292791, 5.3075094, -6.1292791, 5.3075094, -11.4367886, 11.4367886
2: -7.4691033, 4.5115442, -7.4691033, 4.5115442, -11.9806461, 11.9806480
3: -9.0654783, 4.4164257, -9.0654783, 4.4164257, -13.4819021, 13.4819012
4: -8.2032833, 6.3942995, -8.2032833, 6.3942995, -14.5975828, 14.5975819
5: -6.5569053, 5.5308518, -6.5569053, 5.5308518, -12.0877571, 12.0877571
6: -6.6569500, 7.0587082, -6.6569500, 7.0587082, -13.7156572, 13.7156572
7: -8.3428059, 4.1933599, -8.3428059, 4.1933599, -12.5361652, 12.5361652
8: -8.1921329, 6.0312138, -8.1921329, 6.0312138, -14.2233467, 14.2233467
9: -6.3373275, 6.7819953, -6.3373275, 6.7819953, -13.1193199, 13.1193218

Time for backsubstitution: 1.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087835, upper bound: 9.0087846
time: 1.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087835, upper bound: 9.0087846
time: 2.50 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6137094, 5.8590560, -7.6137094, 5.8590560, -13.4727650, 13.4727640
1: -6.1292791, 5.3075094, -6.1292791, 5.3075094, -11.4367886, 11.4367886
2: -7.4691033, 4.5115442, -7.4691033, 4.5115442, -11.9806461, 11.9806480
3: -9.0654783, 4.4164257, -9.0654783, 4.4164257, -13.4819021, 13.4819012
4: -8.2032833, 6.3942995, -8.2032833, 6.3942995, -14.5975828, 14.5975819
5: -6.5569053, 5.5308518, -6.5569053, 5.5308518, -12.0877571, 12.0877571
6: -6.6569500, 7.0587082, -6.6569500, 7.0587082, -13.7156572, 13.7156572
7: -8.3428059, 4.1933599, -8.3428059, 4.1933599, -12.5361652, 12.5361652
8: -8.1921329, 6.0312138, -8.1921329, 6.0312138, -14.2233467, 14.2233467
9: -6.3373275, 6.7819953, -6.3373275, 6.7819953, -13.1193199, 13.1193218

Time for backsubstitution: 1.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087835
time: 2.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087835
time: 7.68 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6137094, 5.8590560, -7.6137094, 5.8590560, -13.4727650, 13.4727640
1: -6.1292791, 5.3075094, -6.1292791, 5.3075094, -11.4367886, 11.4367886
2: -7.4691033, 4.5115442, -7.4691033, 4.5115442, -11.9806461, 11.9806480
3: -9.0654783, 4.4164257, -9.0654783, 4.4164257, -13.4819021, 13.4819012
4: -8.2032833, 6.3942995, -8.2032833, 6.3942995, -14.5975828, 14.5975819
5: -6.5569053, 5.5308518, -6.5569053, 5.5308518, -12.0877571, 12.0877571
6: -6.6569500, 7.0587082, -6.6569500, 7.0587082, -13.7156572, 13.7156572
7: -8.3428059, 4.1933599, -8.3428059, 4.1933599, -12.5361652, 12.5361652
8: -8.1921329, 6.0312138, -8.1921329, 6.0312138, -14.2233467, 14.2233467
9: -6.3373275, 6.7819953, -6.3373275, 6.7819953, -13.1193199, 13.1193218

Time for backsubstitution: 1.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087835
time: 2.09 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087835
time: 2.08 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6137094, 5.8590560, -7.6137094, 5.8590560, -13.4727650, 13.4727640
1: -6.1292791, 5.3075094, -6.1292791, 5.3075094, -11.4367886, 11.4367886
2: -7.4691033, 4.5115442, -7.4691033, 4.5115442, -11.9806461, 11.9806480
3: -9.0654783, 4.4164257, -9.0654783, 4.4164257, -13.4819021, 13.4819012
4: -8.2032833, 6.3942995, -8.2032833, 6.3942995, -14.5975828, 14.5975819
5: -6.5569053, 5.5308518, -6.5569053, 5.5308518, -12.0877571, 12.0877571
6: -6.6569500, 7.0587082, -6.6569500, 7.0587082, -13.7156572, 13.7156572
7: -8.3428059, 4.1933599, -8.3428059, 4.1933599, -12.5361652, 12.5361652
8: -8.1921329, 6.0312138, -8.1921329, 6.0312138, -14.2233467, 14.2233467
9: -6.3373275, 6.7819953, -6.3373275, 6.7819953, -13.1193199, 13.1193218

Time for backsubstitution: 2.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087841, upper bound: 9.0087843
time: 2.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087840, upper bound: 9.0087843
time: 1.86 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6137094, 5.8590560, -7.6137094, 5.8590560, -13.4727650, 13.4727640
1: -6.1292791, 5.3075094, -6.1292791, 5.3075094, -11.4367886, 11.4367886
2: -7.4691033, 4.5115442, -7.4691033, 4.5115442, -11.9806461, 11.9806480
3: -9.0654783, 4.4164257, -9.0654783, 4.4164257, -13.4819021, 13.4819012
4: -8.2032833, 6.3942995, -8.2032833, 6.3942995, -14.5975828, 14.5975819
5: -6.5569053, 5.5308518, -6.5569053, 5.5308518, -12.0877571, 12.0877571
6: -6.6569500, 7.0587082, -6.6569500, 7.0587082, -13.7156572, 13.7156572
7: -8.3428059, 4.1933599, -8.3428059, 4.1933599, -12.5361652, 12.5361652
8: -8.1921329, 6.0312138, -8.1921329, 6.0312138, -14.2233467, 14.2233467
9: -6.3373275, 6.7819953, -6.3373275, 6.7819953, -13.1193199, 13.1193218

Time for backsubstitution: 2.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087841, upper bound: 9.0087843
time: 2.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087841, upper bound: 9.0087843
time: 2.48 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6137094, 5.8590560, -7.6137094, 5.8590560, -13.4727650, 13.4727640
1: -6.1292791, 5.3075094, -6.1292791, 5.3075094, -11.4367886, 11.4367886
2: -7.4691033, 4.5115442, -7.4691033, 4.5115442, -11.9806461, 11.9806480
3: -9.0654783, 4.4164257, -9.0654783, 4.4164257, -13.4819021, 13.4819012
4: -8.2032833, 6.3942995, -8.2032833, 6.3942995, -14.5975828, 14.5975819
5: -6.5569053, 5.5308518, -6.5569053, 5.5308518, -12.0877571, 12.0877571
6: -6.6569500, 7.0587082, -6.6569500, 7.0587082, -13.7156572, 13.7156572
7: -8.3428059, 4.1933599, -8.3428059, 4.1933599, -12.5361652, 12.5361652
8: -8.1921329, 6.0312138, -8.1921329, 6.0312138, -14.2233467, 14.2233467
9: -6.3373275, 6.7819953, -6.3373275, 6.7819953, -13.1193199, 13.1193218

Time for backsubstitution: 1.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087843, upper bound: 9.0087841
time: 3.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087843, upper bound: 9.0087841
time: 1.77 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6137094, 5.8590560, -7.6137094, 5.8590560, -13.4727650, 13.4727640
1: -6.1292791, 5.3075094, -6.1292791, 5.3075094, -11.4367886, 11.4367886
2: -7.4691033, 4.5115442, -7.4691033, 4.5115442, -11.9806461, 11.9806480
3: -9.0654783, 4.4164257, -9.0654783, 4.4164257, -13.4819021, 13.4819012
4: -8.2032833, 6.3942995, -8.2032833, 6.3942995, -14.5975828, 14.5975819
5: -6.5569053, 5.5308518, -6.5569053, 5.5308518, -12.0877571, 12.0877571
6: -6.6569500, 7.0587082, -6.6569500, 7.0587082, -13.7156572, 13.7156572
7: -8.3428059, 4.1933599, -8.3428059, 4.1933599, -12.5361652, 12.5361652
8: -8.1921329, 6.0312138, -8.1921329, 6.0312138, -14.2233467, 14.2233467
9: -6.3373275, 6.7819953, -6.3373275, 6.7819953, -13.1193199, 13.1193218

Time for backsubstitution: 1.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087843, upper bound: 9.0087840
time: 1.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087844, upper bound: 9.0087841
time: 2.75 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6137094, 5.8590560, -7.6137094, 5.8590560, -13.4727650, 13.4727640
1: -6.1292791, 5.3075094, -6.1292791, 5.3075094, -11.4367886, 11.4367886
2: -7.4691033, 4.5115442, -7.4691033, 4.5115442, -11.9806461, 11.9806480
3: -9.0654783, 4.4164257, -9.0654783, 4.4164257, -13.4819021, 13.4819012
4: -8.2032833, 6.3942995, -8.2032833, 6.3942995, -14.5975828, 14.5975819
5: -6.5569053, 5.5308518, -6.5569053, 5.5308518, -12.0877571, 12.0877571
6: -6.6569500, 7.0587082, -6.6569500, 7.0587082, -13.7156572, 13.7156572
7: -8.3428059, 4.1933599, -8.3428059, 4.1933599, -12.5361652, 12.5361652
8: -8.1921329, 6.0312138, -8.1921329, 6.0312138, -14.2233467, 14.2233467
9: -6.3373275, 6.7819953, -6.3373275, 6.7819953, -13.1193199, 13.1193218

Time for backsubstitution: 1.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087835, upper bound: 9.0087846
time: 2.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087835, upper bound: 9.0087846
time: 1.74 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6137094, 5.8590560, -7.6137094, 5.8590560, -13.4727650, 13.4727640
1: -6.1292791, 5.3075094, -6.1292791, 5.3075094, -11.4367886, 11.4367886
2: -7.4691033, 4.5115442, -7.4691033, 4.5115442, -11.9806461, 11.9806480
3: -9.0654783, 4.4164257, -9.0654783, 4.4164257, -13.4819021, 13.4819012
4: -8.2032833, 6.3942995, -8.2032833, 6.3942995, -14.5975828, 14.5975819
5: -6.5569053, 5.5308518, -6.5569053, 5.5308518, -12.0877571, 12.0877571
6: -6.6569500, 7.0587082, -6.6569500, 7.0587082, -13.7156572, 13.7156572
7: -8.3428059, 4.1933599, -8.3428059, 4.1933599, -12.5361652, 12.5361652
8: -8.1921329, 6.0312138, -8.1921329, 6.0312138, -14.2233467, 14.2233467
9: -6.3373275, 6.7819953, -6.3373275, 6.7819953, -13.1193199, 13.1193218

Time for backsubstitution: 1.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087836, upper bound: 9.0087846
time: 3.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087835, upper bound: 9.0087846
time: 2.26 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6137094, 5.8590560, -7.6137094, 5.8590560, -13.4727650, 13.4727640
1: -6.1292791, 5.3075094, -6.1292791, 5.3075094, -11.4367886, 11.4367886
2: -7.4691033, 4.5115442, -7.4691033, 4.5115442, -11.9806461, 11.9806480
3: -9.0654783, 4.4164257, -9.0654783, 4.4164257, -13.4819021, 13.4819012
4: -8.2032833, 6.3942995, -8.2032833, 6.3942995, -14.5975828, 14.5975819
5: -6.5569053, 5.5308518, -6.5569053, 5.5308518, -12.0877571, 12.0877571
6: -6.6569500, 7.0587082, -6.6569500, 7.0587082, -13.7156572, 13.7156572
7: -8.3428059, 4.1933599, -8.3428059, 4.1933599, -12.5361652, 12.5361652
8: -8.1921329, 6.0312138, -8.1921329, 6.0312138, -14.2233467, 14.2233467
9: -6.3373275, 6.7819953, -6.3373275, 6.7819953, -13.1193199, 13.1193218

Time for backsubstitution: 1.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087835
time: 6.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087836
time: 2.91 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6137094, 5.8590560, -7.6137094, 5.8590560, -13.4727650, 13.4727640
1: -6.1292791, 5.3075094, -6.1292791, 5.3075094, -11.4367886, 11.4367886
2: -7.4691033, 4.5115442, -7.4691033, 4.5115442, -11.9806461, 11.9806480
3: -9.0654783, 4.4164257, -9.0654783, 4.4164257, -13.4819021, 13.4819012
4: -8.2032833, 6.3942995, -8.2032833, 6.3942995, -14.5975828, 14.5975819
5: -6.5569053, 5.5308518, -6.5569053, 5.5308518, -12.0877571, 12.0877571
6: -6.6569500, 7.0587082, -6.6569500, 7.0587082, -13.7156572, 13.7156572
7: -8.3428059, 4.1933599, -8.3428059, 4.1933599, -12.5361652, 12.5361652
8: -8.1921329, 6.0312138, -8.1921329, 6.0312138, -14.2233467, 14.2233467
9: -6.3373275, 6.7819953, -6.3373275, 6.7819953, -13.1193199, 13.1193218

Time for backsubstitution: 2.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087835
time: 2.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087835
time: 2.40 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6137094, 5.8590560, -7.6137094, 5.8590560, -13.4727650, 13.4727640
1: -6.1292791, 5.3075094, -6.1292791, 5.3075094, -11.4367886, 11.4367886
2: -7.4691033, 4.5115442, -7.4691033, 4.5115442, -11.9806461, 11.9806480
3: -9.0654783, 4.4164257, -9.0654783, 4.4164257, -13.4819021, 13.4819012
4: -8.2032833, 6.3942995, -8.2032833, 6.3942995, -14.5975828, 14.5975819
5: -6.5569053, 5.5308518, -6.5569053, 5.5308518, -12.0877571, 12.0877571
6: -6.6569500, 7.0587082, -6.6569500, 7.0587082, -13.7156572, 13.7156572
7: -8.3428059, 4.1933599, -8.3428059, 4.1933599, -12.5361652, 12.5361652
8: -8.1921329, 6.0312138, -8.1921329, 6.0312138, -14.2233467, 14.2233467
9: -6.3373275, 6.7819953, -6.3373275, 6.7819953, -13.1193199, 13.1193218

Time for backsubstitution: 1.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087841, upper bound: 9.0087844
time: 3.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087840, upper bound: 9.0087843
time: 2.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6137094, 5.8590560, -7.6137094, 5.8590560, -13.4727650, 13.4727640
1: -6.1292791, 5.3075094, -6.1292791, 5.3075094, -11.4367886, 11.4367886
2: -7.4691033, 4.5115442, -7.4691033, 4.5115442, -11.9806461, 11.9806480
3: -9.0654783, 4.4164257, -9.0654783, 4.4164257, -13.4819021, 13.4819012
4: -8.2032833, 6.3942995, -8.2032833, 6.3942995, -14.5975828, 14.5975819
5: -6.5569053, 5.5308518, -6.5569053, 5.5308518, -12.0877571, 12.0877571
6: -6.6569500, 7.0587082, -6.6569500, 7.0587082, -13.7156572, 13.7156572
7: -8.3428059, 4.1933599, -8.3428059, 4.1933599, -12.5361652, 12.5361652
8: -8.1921329, 6.0312138, -8.1921329, 6.0312138, -14.2233467, 14.2233467
9: -6.3373275, 6.7819953, -6.3373275, 6.7819953, -13.1193199, 13.1193218

Time for backsubstitution: 1.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087841, upper bound: 9.0087843
time: 1.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087841, upper bound: 9.0087843
time: 2.72 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6137094, 5.8590560, -7.6137094, 5.8590560, -13.4727650, 13.4727640
1: -6.1292791, 5.3075094, -6.1292791, 5.3075094, -11.4367886, 11.4367886
2: -7.4691033, 4.5115442, -7.4691033, 4.5115442, -11.9806461, 11.9806480
3: -9.0654783, 4.4164257, -9.0654783, 4.4164257, -13.4819021, 13.4819012
4: -8.2032833, 6.3942995, -8.2032833, 6.3942995, -14.5975828, 14.5975819
5: -6.5569053, 5.5308518, -6.5569053, 5.5308518, -12.0877571, 12.0877571
6: -6.6569500, 7.0587082, -6.6569500, 7.0587082, -13.7156572, 13.7156572
7: -8.3428059, 4.1933599, -8.3428059, 4.1933599, -12.5361652, 12.5361652
8: -8.1921329, 6.0312138, -8.1921329, 6.0312138, -14.2233467, 14.2233467
9: -6.3373275, 6.7819953, -6.3373275, 6.7819953, -13.1193199, 13.1193218

Time for backsubstitution: 1.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087843, upper bound: 9.0087841
time: 2.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087843, upper bound: 9.0087841
time: 1.70 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6137094, 5.8590560, -7.6137094, 5.8590560, -13.4727650, 13.4727640
1: -6.1292791, 5.3075094, -6.1292791, 5.3075094, -11.4367886, 11.4367886
2: -7.4691033, 4.5115442, -7.4691033, 4.5115442, -11.9806461, 11.9806480
3: -9.0654783, 4.4164257, -9.0654783, 4.4164257, -13.4819021, 13.4819012
4: -8.2032833, 6.3942995, -8.2032833, 6.3942995, -14.5975828, 14.5975819
5: -6.5569053, 5.5308518, -6.5569053, 5.5308518, -12.0877571, 12.0877571
6: -6.6569500, 7.0587082, -6.6569500, 7.0587082, -13.7156572, 13.7156572
7: -8.3428059, 4.1933599, -8.3428059, 4.1933599, -12.5361652, 12.5361652
8: -8.1921329, 6.0312138, -8.1921329, 6.0312138, -14.2233467, 14.2233467
9: -6.3373275, 6.7819953, -6.3373275, 6.7819953, -13.1193199, 13.1193218

Time for backsubstitution: 1.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087843, upper bound: 9.0087841
time: 1.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087843, upper bound: 9.0087841
time: 2.31 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6137094, 5.8590560, -7.6137094, 5.8590560, -13.4727650, 13.4727640
1: -6.1292791, 5.3075094, -6.1292791, 5.3075094, -11.4367886, 11.4367886
2: -7.4691033, 4.5115442, -7.4691033, 4.5115442, -11.9806461, 11.9806480
3: -9.0654783, 4.4164257, -9.0654783, 4.4164257, -13.4819021, 13.4819012
4: -8.2032833, 6.3942995, -8.2032833, 6.3942995, -14.5975828, 14.5975819
5: -6.5569053, 5.5308518, -6.5569053, 5.5308518, -12.0877571, 12.0877571
6: -6.6569500, 7.0587082, -6.6569500, 7.0587082, -13.7156572, 13.7156572
7: -8.3428059, 4.1933599, -8.3428059, 4.1933599, -12.5361652, 12.5361652
8: -8.1921329, 6.0312138, -8.1921329, 6.0312138, -14.2233467, 14.2233467
9: -6.3373275, 6.7819953, -6.3373275, 6.7819953, -13.1193199, 13.1193218

Time for backsubstitution: 1.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087835, upper bound: 9.0087846
time: 2.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087835, upper bound: 9.0087846
time: 1.67 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6137094, 5.8590560, -7.6137094, 5.8590560, -13.4727650, 13.4727640
1: -6.1292791, 5.3075094, -6.1292791, 5.3075094, -11.4367886, 11.4367886
2: -7.4691033, 4.5115442, -7.4691033, 4.5115442, -11.9806461, 11.9806480
3: -9.0654783, 4.4164257, -9.0654783, 4.4164257, -13.4819021, 13.4819012
4: -8.2032833, 6.3942995, -8.2032833, 6.3942995, -14.5975828, 14.5975819
5: -6.5569053, 5.5308518, -6.5569053, 5.5308518, -12.0877571, 12.0877571
6: -6.6569500, 7.0587082, -6.6569500, 7.0587082, -13.7156572, 13.7156572
7: -8.3428059, 4.1933599, -8.3428059, 4.1933599, -12.5361652, 12.5361652
8: -8.1921329, 6.0312138, -8.1921329, 6.0312138, -14.2233467, 14.2233467
9: -6.3373275, 6.7819953, -6.3373275, 6.7819953, -13.1193199, 13.1193218

Time for backsubstitution: 2.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087835, upper bound: 9.0087846
time: 1.90 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087835, upper bound: 9.0087846
time: 2.42 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6137094, 5.8590560, -7.6137094, 5.8590560, -13.4727650, 13.4727640
1: -6.1292791, 5.3075094, -6.1292791, 5.3075094, -11.4367886, 11.4367886
2: -7.4691033, 4.5115442, -7.4691033, 4.5115442, -11.9806461, 11.9806480
3: -9.0654783, 4.4164257, -9.0654783, 4.4164257, -13.4819021, 13.4819012
4: -8.2032833, 6.3942995, -8.2032833, 6.3942995, -14.5975828, 14.5975819
5: -6.5569053, 5.5308518, -6.5569053, 5.5308518, -12.0877571, 12.0877571
6: -6.6569500, 7.0587082, -6.6569500, 7.0587082, -13.7156572, 13.7156572
7: -8.3428059, 4.1933599, -8.3428059, 4.1933599, -12.5361652, 12.5361652
8: -8.1921329, 6.0312138, -8.1921329, 6.0312138, -14.2233467, 14.2233467
9: -6.3373275, 6.7819953, -6.3373275, 6.7819953, -13.1193199, 13.1193218

Time for backsubstitution: 2.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087835
time: 2.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087835
time: 7.93 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6137094, 5.8590560, -7.6137094, 5.8590560, -13.4727650, 13.4727640
1: -6.1292791, 5.3075094, -6.1292791, 5.3075094, -11.4367886, 11.4367886
2: -7.4691033, 4.5115442, -7.4691033, 4.5115442, -11.9806461, 11.9806480
3: -9.0654783, 4.4164257, -9.0654783, 4.4164257, -13.4819021, 13.4819012
4: -8.2032833, 6.3942995, -8.2032833, 6.3942995, -14.5975828, 14.5975819
5: -6.5569053, 5.5308518, -6.5569053, 5.5308518, -12.0877571, 12.0877571
6: -6.6569500, 7.0587082, -6.6569500, 7.0587082, -13.7156572, 13.7156572
7: -8.3428059, 4.1933599, -8.3428059, 4.1933599, -12.5361652, 12.5361652
8: -8.1921329, 6.0312138, -8.1921329, 6.0312138, -14.2233467, 14.2233467
9: -6.3373275, 6.7819953, -6.3373275, 6.7819953, -13.1193199, 13.1193218

Time for backsubstitution: 1.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087835
time: 2.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087835
time: 2.13 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6137094, 5.8590560, -7.6137094, 5.8590560, -13.4727650, 13.4727640
1: -6.1292791, 5.3075094, -6.1292791, 5.3075094, -11.4367886, 11.4367886
2: -7.4691033, 4.5115442, -7.4691033, 4.5115442, -11.9806461, 11.9806480
3: -9.0654783, 4.4164257, -9.0654783, 4.4164257, -13.4819021, 13.4819012
4: -8.2032833, 6.3942995, -8.2032833, 6.3942995, -14.5975828, 14.5975819
5: -6.5569053, 5.5308518, -6.5569053, 5.5308518, -12.0877571, 12.0877571
6: -6.6569500, 7.0587082, -6.6569500, 7.0587082, -13.7156572, 13.7156572
7: -8.3428059, 4.1933599, -8.3428059, 4.1933599, -12.5361652, 12.5361652
8: -8.1921329, 6.0312138, -8.1921329, 6.0312138, -14.2233467, 14.2233467
9: -6.3373275, 6.7819953, -6.3373275, 6.7819953, -13.1193199, 13.1193218

Time for backsubstitution: 2.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087841, upper bound: 9.0087843
time: 2.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087840, upper bound: 9.0087843
time: 1.66 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6137094, 5.8590560, -7.6137094, 5.8590560, -13.4727650, 13.4727640
1: -6.1292791, 5.3075094, -6.1292791, 5.3075094, -11.4367886, 11.4367886
2: -7.4691033, 4.5115442, -7.4691033, 4.5115442, -11.9806461, 11.9806480
3: -9.0654783, 4.4164257, -9.0654783, 4.4164257, -13.4819021, 13.4819012
4: -8.2032833, 6.3942995, -8.2032833, 6.3942995, -14.5975828, 14.5975819
5: -6.5569053, 5.5308518, -6.5569053, 5.5308518, -12.0877571, 12.0877571
6: -6.6569500, 7.0587082, -6.6569500, 7.0587082, -13.7156572, 13.7156572
7: -8.3428059, 4.1933599, -8.3428059, 4.1933599, -12.5361652, 12.5361652
8: -8.1921329, 6.0312138, -8.1921329, 6.0312138, -14.2233467, 14.2233467
9: -6.3373275, 6.7819953, -6.3373275, 6.7819953, -13.1193199, 13.1193218

Time for backsubstitution: 1.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087841, upper bound: 9.0087843
time: 2.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087841, upper bound: 9.0087843
time: 2.42 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6137094, 5.8590560, -7.6137094, 5.8590560, -13.4727650, 13.4727640
1: -6.1292791, 5.3075094, -6.1292791, 5.3075094, -11.4367886, 11.4367886
2: -7.4691033, 4.5115442, -7.4691033, 4.5115442, -11.9806461, 11.9806480
3: -9.0654783, 4.4164257, -9.0654783, 4.4164257, -13.4819021, 13.4819012
4: -8.2032833, 6.3942995, -8.2032833, 6.3942995, -14.5975828, 14.5975819
5: -6.5569053, 5.5308518, -6.5569053, 5.5308518, -12.0877571, 12.0877571
6: -6.6569500, 7.0587082, -6.6569500, 7.0587082, -13.7156572, 13.7156572
7: -8.3428059, 4.1933599, -8.3428059, 4.1933599, -12.5361652, 12.5361652
8: -8.1921329, 6.0312138, -8.1921329, 6.0312138, -14.2233467, 14.2233467
9: -6.3373275, 6.7819953, -6.3373275, 6.7819953, -13.1193199, 13.1193218

Time for backsubstitution: 1.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087843, upper bound: 9.0087841
time: 3.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087843, upper bound: 9.0087841
time: 1.78 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6137094, 5.8590560, -7.6137094, 5.8590560, -13.4727650, 13.4727640
1: -6.1292791, 5.3075094, -6.1292791, 5.3075094, -11.4367886, 11.4367886
2: -7.4691033, 4.5115442, -7.4691033, 4.5115442, -11.9806461, 11.9806480
3: -9.0654783, 4.4164257, -9.0654783, 4.4164257, -13.4819021, 13.4819012
4: -8.2032833, 6.3942995, -8.2032833, 6.3942995, -14.5975828, 14.5975819
5: -6.5569053, 5.5308518, -6.5569053, 5.5308518, -12.0877571, 12.0877571
6: -6.6569500, 7.0587082, -6.6569500, 7.0587082, -13.7156572, 13.7156572
7: -8.3428059, 4.1933599, -8.3428059, 4.1933599, -12.5361652, 12.5361652
8: -8.1921329, 6.0312138, -8.1921329, 6.0312138, -14.2233467, 14.2233467
9: -6.3373275, 6.7819953, -6.3373275, 6.7819953, -13.1193199, 13.1193218

Time for backsubstitution: 1.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087843, upper bound: 9.0087840
time: 1.90 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087844, upper bound: 9.0087841
time: 2.76 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6137094, 5.8590560, -7.6137094, 5.8590560, -13.4727650, 13.4727640
1: -6.1292791, 5.3075094, -6.1292791, 5.3075094, -11.4367886, 11.4367886
2: -7.4691033, 4.5115442, -7.4691033, 4.5115442, -11.9806461, 11.9806480
3: -9.0654783, 4.4164257, -9.0654783, 4.4164257, -13.4819021, 13.4819012
4: -8.2032833, 6.3942995, -8.2032833, 6.3942995, -14.5975828, 14.5975819
5: -6.5569053, 5.5308518, -6.5569053, 5.5308518, -12.0877571, 12.0877571
6: -6.6569500, 7.0587082, -6.6569500, 7.0587082, -13.7156572, 13.7156572
7: -8.3428059, 4.1933599, -8.3428059, 4.1933599, -12.5361652, 12.5361652
8: -8.1921329, 6.0312138, -8.1921329, 6.0312138, -14.2233467, 14.2233467
9: -6.3373275, 6.7819953, -6.3373275, 6.7819953, -13.1193199, 13.1193218

Time for backsubstitution: 1.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087835, upper bound: 9.0087846
time: 2.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087835, upper bound: 9.0087846
time: 1.71 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6137094, 5.8590560, -7.6137094, 5.8590560, -13.4727650, 13.4727640
1: -6.1292791, 5.3075094, -6.1292791, 5.3075094, -11.4367886, 11.4367886
2: -7.4691033, 4.5115442, -7.4691033, 4.5115442, -11.9806461, 11.9806480
3: -9.0654783, 4.4164257, -9.0654783, 4.4164257, -13.4819021, 13.4819012
4: -8.2032833, 6.3942995, -8.2032833, 6.3942995, -14.5975828, 14.5975819
5: -6.5569053, 5.5308518, -6.5569053, 5.5308518, -12.0877571, 12.0877571
6: -6.6569500, 7.0587082, -6.6569500, 7.0587082, -13.7156572, 13.7156572
7: -8.3428059, 4.1933599, -8.3428059, 4.1933599, -12.5361652, 12.5361652
8: -8.1921329, 6.0312138, -8.1921329, 6.0312138, -14.2233467, 14.2233467
9: -6.3373275, 6.7819953, -6.3373275, 6.7819953, -13.1193199, 13.1193218

Time for backsubstitution: 1.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087836, upper bound: 9.0087846
time: 4.02 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087835, upper bound: 9.0087846
time: 2.57 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 8.74 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087835
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087836
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087835
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087835
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -9.0087841, upper bound: 9.0087844
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -9.0087840, upper bound: 9.0087843
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -9.0087841, upper bound: 9.0087843
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -9.0087841, upper bound: 9.0087843
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -9.0087843, upper bound: 9.0087841
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -9.0087843, upper bound: 9.0087841
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -9.0087843, upper bound: 9.0087841
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -9.0087843, upper bound: 9.0087841
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -9.0087835, upper bound: 9.0087846
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -9.0087835, upper bound: 9.0087846
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -9.0087835, upper bound: 9.0087846
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -9.0087835, upper bound: 9.0087846
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087835
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087835
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087835
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087835
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -9.0087841, upper bound: 9.0087843
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -9.0087840, upper bound: 9.0087843
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -9.0087841, upper bound: 9.0087843
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -9.0087841, upper bound: 9.0087843
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -9.0087843, upper bound: 9.0087841
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -9.0087843, upper bound: 9.0087841
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -9.0087843, upper bound: 9.0087840
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -9.0087844, upper bound: 9.0087841
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -9.0087835, upper bound: 9.0087846
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -9.0087835, upper bound: 9.0087846
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -9.0087836, upper bound: 9.0087846
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -9.0087835, upper bound: 9.0087846
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087835
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087836
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087835
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087835
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -9.0087841, upper bound: 9.0087844
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -9.0087840, upper bound: 9.0087843
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -9.0087841, upper bound: 9.0087843
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -9.0087841, upper bound: 9.0087843
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -9.0087843, upper bound: 9.0087841
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -9.0087843, upper bound: 9.0087841
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -9.0087843, upper bound: 9.0087841
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -9.0087843, upper bound: 9.0087841
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -9.0087835, upper bound: 9.0087846
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -9.0087835, upper bound: 9.0087846
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -9.0087835, upper bound: 9.0087846
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -9.0087835, upper bound: 9.0087846
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087835
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087835
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087835
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087835
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -9.0087841, upper bound: 9.0087843
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -9.0087840, upper bound: 9.0087843
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -9.0087841, upper bound: 9.0087843
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -9.0087841, upper bound: 9.0087843
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -9.0087843, upper bound: 9.0087841
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -9.0087843, upper bound: 9.0087841
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -9.0087843, upper bound: 9.0087840
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -9.0087844, upper bound: 9.0087841
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -9.0087835, upper bound: 9.0087846
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -9.0087835, upper bound: 9.0087846
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -9.0087836, upper bound: 9.0087846
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -9.0087835, upper bound: 9.0087846

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6137094, 5.8590560, -7.6137094, 5.8590560, -13.4727650, 13.4727640
1: -6.1292791, 5.3075094, -6.1292791, 5.3075094, -11.4367886, 11.4367886
2: -7.4691033, 4.5115442, -7.4691033, 4.5115442, -11.9806461, 11.9806480
3: -9.0654783, 4.4164257, -9.0654783, 4.4164257, -13.4819021, 13.4819012
4: -8.2032833, 6.3942995, -8.2032833, 6.3942995, -14.5975828, 14.5975819
5: -6.5569053, 5.5308518, -6.5569053, 5.5308518, -12.0877571, 12.0877571
6: -6.6569500, 7.0587082, -6.6569500, 7.0587082, -13.7156572, 13.7156572
7: -8.3428059, 4.1933599, -8.3428059, 4.1933599, -12.5361652, 12.5361652
8: -8.1921329, 6.0312138, -8.1921329, 6.0312138, -14.2233467, 14.2233467
9: -6.3373275, 6.7819953, -6.3373275, 6.7819953, -13.1193199, 13.1193218

Time for backsubstitution: 2.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.28 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087835
time: 2.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087845, upper bound: 9.0087835
time: 1.76 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6137094, 5.8590560, -7.6137094, 5.8590560, -13.4727650, 13.4727640
1: -6.1292791, 5.3075094, -6.1292791, 5.3075094, -11.4367886, 11.4367886
2: -7.4691033, 4.5115442, -7.4691033, 4.5115442, -11.9806461, 11.9806480
3: -9.0654783, 4.4164257, -9.0654783, 4.4164257, -13.4819021, 13.4819012
4: -8.2032833, 6.3942995, -8.2032833, 6.3942995, -14.5975828, 14.5975819
5: -6.5569053, 5.5308518, -6.5569053, 5.5308518, -12.0877571, 12.0877571
6: -6.6569500, 7.0587082, -6.6569500, 7.0587082, -13.7156572, 13.7156572
7: -8.3428059, 4.1933599, -8.3428059, 4.1933599, -12.5361652, 12.5361652
8: -8.1921329, 6.0312138, -8.1921329, 6.0312138, -14.2233467, 14.2233467
9: -6.3373275, 6.7819953, -6.3373275, 6.7819953, -13.1193199, 13.1193218

Time for backsubstitution: 2.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087835
time: 2.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087845, upper bound: 9.0087835
time: 3.29 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6137094, 5.8590560, -7.6137094, 5.8590560, -13.4727650, 13.4727640
1: -6.1292791, 5.3075094, -6.1292791, 5.3075094, -11.4367886, 11.4367886
2: -7.4691033, 4.5115442, -7.4691033, 4.5115442, -11.9806461, 11.9806480
3: -9.0654783, 4.4164257, -9.0654783, 4.4164257, -13.4819021, 13.4819012
4: -8.2032833, 6.3942995, -8.2032833, 6.3942995, -14.5975828, 14.5975819
5: -6.5569053, 5.5308518, -6.5569053, 5.5308518, -12.0877571, 12.0877571
6: -6.6569500, 7.0587082, -6.6569500, 7.0587082, -13.7156572, 13.7156572
7: -8.3428059, 4.1933599, -8.3428059, 4.1933599, -12.5361652, 12.5361652
8: -8.1921329, 6.0312138, -8.1921329, 6.0312138, -14.2233467, 14.2233467
9: -6.3373275, 6.7819953, -6.3373275, 6.7819953, -13.1193199, 13.1193218

Time for backsubstitution: 2.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087835
time: 1.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087845, upper bound: 9.0087835
time: 4.62 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6137094, 5.8590560, -7.6137094, 5.8590560, -13.4727650, 13.4727640
1: -6.1292791, 5.3075094, -6.1292791, 5.3075094, -11.4367886, 11.4367886
2: -7.4691033, 4.5115442, -7.4691033, 4.5115442, -11.9806461, 11.9806480
3: -9.0654783, 4.4164257, -9.0654783, 4.4164257, -13.4819021, 13.4819012
4: -8.2032833, 6.3942995, -8.2032833, 6.3942995, -14.5975828, 14.5975819
5: -6.5569053, 5.5308518, -6.5569053, 5.5308518, -12.0877571, 12.0877571
6: -6.6569500, 7.0587082, -6.6569500, 7.0587082, -13.7156572, 13.7156572
7: -8.3428059, 4.1933599, -8.3428059, 4.1933599, -12.5361652, 12.5361652
8: -8.1921329, 6.0312138, -8.1921329, 6.0312138, -14.2233467, 14.2233467
9: -6.3373275, 6.7819953, -6.3373275, 6.7819953, -13.1193199, 13.1193218

Time for backsubstitution: 2.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087834
time: 2.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087845, upper bound: 9.0087835
time: 1.77 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6137094, 5.8590560, -7.6137094, 5.8590560, -13.4727650, 13.4727640
1: -6.1292791, 5.3075094, -6.1292791, 5.3075094, -11.4367886, 11.4367886
2: -7.4691033, 4.5115442, -7.4691033, 4.5115442, -11.9806461, 11.9806480
3: -9.0654783, 4.4164257, -9.0654783, 4.4164257, -13.4819021, 13.4819012
4: -8.2032833, 6.3942995, -8.2032833, 6.3942995, -14.5975828, 14.5975819
5: -6.5569053, 5.5308518, -6.5569053, 5.5308518, -12.0877571, 12.0877571
6: -6.6569500, 7.0587082, -6.6569500, 7.0587082, -13.7156572, 13.7156572
7: -8.3428059, 4.1933599, -8.3428059, 4.1933599, -12.5361652, 12.5361652
8: -8.1921329, 6.0312138, -8.1921329, 6.0312138, -14.2233467, 14.2233467
9: -6.3373275, 6.7819953, -6.3373275, 6.7819953, -13.1193199, 13.1193218

Time for backsubstitution: 1.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087841, upper bound: 9.0087843
time: 1.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087840, upper bound: 9.0087844
time: 1.74 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6137094, 5.8590560, -7.6137094, 5.8590560, -13.4727650, 13.4727640
1: -6.1292791, 5.3075094, -6.1292791, 5.3075094, -11.4367886, 11.4367886
2: -7.4691033, 4.5115442, -7.4691033, 4.5115442, -11.9806461, 11.9806480
3: -9.0654783, 4.4164257, -9.0654783, 4.4164257, -13.4819021, 13.4819012
4: -8.2032833, 6.3942995, -8.2032833, 6.3942995, -14.5975828, 14.5975819
5: -6.5569053, 5.5308518, -6.5569053, 5.5308518, -12.0877571, 12.0877571
6: -6.6569500, 7.0587082, -6.6569500, 7.0587082, -13.7156572, 13.7156572
7: -8.3428059, 4.1933599, -8.3428059, 4.1933599, -12.5361652, 12.5361652
8: -8.1921329, 6.0312138, -8.1921329, 6.0312138, -14.2233467, 14.2233467
9: -6.3373275, 6.7819953, -6.3373275, 6.7819953, -13.1193199, 13.1193218

Time for backsubstitution: 1.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087840, upper bound: 9.0087842
time: 1.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087840, upper bound: 9.0087843
time: 2.46 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6137094, 5.8590560, -7.6137094, 5.8590560, -13.4727650, 13.4727640
1: -6.1292791, 5.3075094, -6.1292791, 5.3075094, -11.4367886, 11.4367886
2: -7.4691033, 4.5115442, -7.4691033, 4.5115442, -11.9806461, 11.9806480
3: -9.0654783, 4.4164257, -9.0654783, 4.4164257, -13.4819021, 13.4819012
4: -8.2032833, 6.3942995, -8.2032833, 6.3942995, -14.5975828, 14.5975819
5: -6.5569053, 5.5308518, -6.5569053, 5.5308518, -12.0877571, 12.0877571
6: -6.6569500, 7.0587082, -6.6569500, 7.0587082, -13.7156572, 13.7156572
7: -8.3428059, 4.1933599, -8.3428059, 4.1933599, -12.5361652, 12.5361652
8: -8.1921329, 6.0312138, -8.1921329, 6.0312138, -14.2233467, 14.2233467
9: -6.3373275, 6.7819953, -6.3373275, 6.7819953, -13.1193199, 13.1193218

Time for backsubstitution: 1.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087841, upper bound: 9.0087842
time: 1.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087840, upper bound: 9.0087843
time: 4.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6137094, 5.8590560, -7.6137094, 5.8590560, -13.4727650, 13.4727640
1: -6.1292791, 5.3075094, -6.1292791, 5.3075094, -11.4367886, 11.4367886
2: -7.4691033, 4.5115442, -7.4691033, 4.5115442, -11.9806461, 11.9806480
3: -9.0654783, 4.4164257, -9.0654783, 4.4164257, -13.4819021, 13.4819012
4: -8.2032833, 6.3942995, -8.2032833, 6.3942995, -14.5975828, 14.5975819
5: -6.5569053, 5.5308518, -6.5569053, 5.5308518, -12.0877571, 12.0877571
6: -6.6569500, 7.0587082, -6.6569500, 7.0587082, -13.7156572, 13.7156572
7: -8.3428059, 4.1933599, -8.3428059, 4.1933599, -12.5361652, 12.5361652
8: -8.1921329, 6.0312138, -8.1921329, 6.0312138, -14.2233467, 14.2233467
9: -6.3373275, 6.7819953, -6.3373275, 6.7819953, -13.1193199, 13.1193218

Time for backsubstitution: 1.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087841, upper bound: 9.0087842
time: 2.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087840, upper bound: 9.0087843
time: 2.45 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6137094, 5.8590560, -7.6137094, 5.8590560, -13.4727650, 13.4727640
1: -6.1292791, 5.3075094, -6.1292791, 5.3075094, -11.4367886, 11.4367886
2: -7.4691033, 4.5115442, -7.4691033, 4.5115442, -11.9806461, 11.9806480
3: -9.0654783, 4.4164257, -9.0654783, 4.4164257, -13.4819021, 13.4819012
4: -8.2032833, 6.3942995, -8.2032833, 6.3942995, -14.5975828, 14.5975819
5: -6.5569053, 5.5308518, -6.5569053, 5.5308518, -12.0877571, 12.0877571
6: -6.6569500, 7.0587082, -6.6569500, 7.0587082, -13.7156572, 13.7156572
7: -8.3428059, 4.1933599, -8.3428059, 4.1933599, -12.5361652, 12.5361652
8: -8.1921329, 6.0312138, -8.1921329, 6.0312138, -14.2233467, 14.2233467
9: -6.3373275, 6.7819953, -6.3373275, 6.7819953, -13.1193199, 13.1193218

Time for backsubstitution: 1.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087843, upper bound: 9.0087840
time: 3.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087841, upper bound: 9.0087841
time: 1.71 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6137094, 5.8590560, -7.6137094, 5.8590560, -13.4727650, 13.4727640
1: -6.1292791, 5.3075094, -6.1292791, 5.3075094, -11.4367886, 11.4367886
2: -7.4691033, 4.5115442, -7.4691033, 4.5115442, -11.9806461, 11.9806480
3: -9.0654783, 4.4164257, -9.0654783, 4.4164257, -13.4819021, 13.4819012
4: -8.2032833, 6.3942995, -8.2032833, 6.3942995, -14.5975828, 14.5975819
5: -6.5569053, 5.5308518, -6.5569053, 5.5308518, -12.0877571, 12.0877571
6: -6.6569500, 7.0587082, -6.6569500, 7.0587082, -13.7156572, 13.7156572
7: -8.3428059, 4.1933599, -8.3428059, 4.1933599, -12.5361652, 12.5361652
8: -8.1921329, 6.0312138, -8.1921329, 6.0312138, -14.2233467, 14.2233467
9: -6.3373275, 6.7819953, -6.3373275, 6.7819953, -13.1193199, 13.1193218

Time for backsubstitution: 1.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087843, upper bound: 9.0087840
time: 2.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087842, upper bound: 9.0087841
time: 2.67 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6137094, 5.8590560, -7.6137094, 5.8590560, -13.4727650, 13.4727640
1: -6.1292791, 5.3075094, -6.1292791, 5.3075094, -11.4367886, 11.4367886
2: -7.4691033, 4.5115442, -7.4691033, 4.5115442, -11.9806461, 11.9806480
3: -9.0654783, 4.4164257, -9.0654783, 4.4164257, -13.4819021, 13.4819012
4: -8.2032833, 6.3942995, -8.2032833, 6.3942995, -14.5975828, 14.5975819
5: -6.5569053, 5.5308518, -6.5569053, 5.5308518, -12.0877571, 12.0877571
6: -6.6569500, 7.0587082, -6.6569500, 7.0587082, -13.7156572, 13.7156572
7: -8.3428059, 4.1933599, -8.3428059, 4.1933599, -12.5361652, 12.5361652
8: -8.1921329, 6.0312138, -8.1921329, 6.0312138, -14.2233467, 14.2233467
9: -6.3373275, 6.7819953, -6.3373275, 6.7819953, -13.1193199, 13.1193218

Time for backsubstitution: 1.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087843, upper bound: 9.0087840
time: 1.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087841, upper bound: 9.0087840
time: 4.92 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6137094, 5.8590560, -7.6137094, 5.8590560, -13.4727650, 13.4727640
1: -6.1292791, 5.3075094, -6.1292791, 5.3075094, -11.4367886, 11.4367886
2: -7.4691033, 4.5115442, -7.4691033, 4.5115442, -11.9806461, 11.9806480
3: -9.0654783, 4.4164257, -9.0654783, 4.4164257, -13.4819021, 13.4819012
4: -8.2032833, 6.3942995, -8.2032833, 6.3942995, -14.5975828, 14.5975819
5: -6.5569053, 5.5308518, -6.5569053, 5.5308518, -12.0877571, 12.0877571
6: -6.6569500, 7.0587082, -6.6569500, 7.0587082, -13.7156572, 13.7156572
7: -8.3428059, 4.1933599, -8.3428059, 4.1933599, -12.5361652, 12.5361652
8: -8.1921329, 6.0312138, -8.1921329, 6.0312138, -14.2233467, 14.2233467
9: -6.3373275, 6.7819953, -6.3373275, 6.7819953, -13.1193199, 13.1193218

Time for backsubstitution: 1.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087843, upper bound: 9.0087840
time: 1.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087842, upper bound: 9.0087841
time: 3.29 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6137094, 5.8590560, -7.6137094, 5.8590560, -13.4727650, 13.4727640
1: -6.1292791, 5.3075094, -6.1292791, 5.3075094, -11.4367886, 11.4367886
2: -7.4691033, 4.5115442, -7.4691033, 4.5115442, -11.9806461, 11.9806480
3: -9.0654783, 4.4164257, -9.0654783, 4.4164257, -13.4819021, 13.4819012
4: -8.2032833, 6.3942995, -8.2032833, 6.3942995, -14.5975828, 14.5975819
5: -6.5569053, 5.5308518, -6.5569053, 5.5308518, -12.0877571, 12.0877571
6: -6.6569500, 7.0587082, -6.6569500, 7.0587082, -13.7156572, 13.7156572
7: -8.3428059, 4.1933599, -8.3428059, 4.1933599, -12.5361652, 12.5361652
8: -8.1921329, 6.0312138, -8.1921329, 6.0312138, -14.2233467, 14.2233467
9: -6.3373275, 6.7819953, -6.3373275, 6.7819953, -13.1193199, 13.1193218

Time for backsubstitution: 1.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.27 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087835, upper bound: 9.0087845
time: 2.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087834, upper bound: 9.0087846
time: 1.91 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6137094, 5.8590560, -7.6137094, 5.8590560, -13.4727650, 13.4727640
1: -6.1292791, 5.3075094, -6.1292791, 5.3075094, -11.4367886, 11.4367886
2: -7.4691033, 4.5115442, -7.4691033, 4.5115442, -11.9806461, 11.9806480
3: -9.0654783, 4.4164257, -9.0654783, 4.4164257, -13.4819021, 13.4819012
4: -8.2032833, 6.3942995, -8.2032833, 6.3942995, -14.5975828, 14.5975819
5: -6.5569053, 5.5308518, -6.5569053, 5.5308518, -12.0877571, 12.0877571
6: -6.6569500, 7.0587082, -6.6569500, 7.0587082, -13.7156572, 13.7156572
7: -8.3428059, 4.1933599, -8.3428059, 4.1933599, -12.5361652, 12.5361652
8: -8.1921329, 6.0312138, -8.1921329, 6.0312138, -14.2233467, 14.2233467
9: -6.3373275, 6.7819953, -6.3373275, 6.7819953, -13.1193199, 13.1193218

Time for backsubstitution: 1.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087835, upper bound: 9.0087845
time: 1.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087834, upper bound: 9.0087846
time: 1.93 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6137094, 5.8590560, -7.6137094, 5.8590560, -13.4727650, 13.4727640
1: -6.1292791, 5.3075094, -6.1292791, 5.3075094, -11.4367886, 11.4367886
2: -7.4691033, 4.5115442, -7.4691033, 4.5115442, -11.9806461, 11.9806480
3: -9.0654783, 4.4164257, -9.0654783, 4.4164257, -13.4819021, 13.4819012
4: -8.2032833, 6.3942995, -8.2032833, 6.3942995, -14.5975828, 14.5975819
5: -6.5569053, 5.5308518, -6.5569053, 5.5308518, -12.0877571, 12.0877571
6: -6.6569500, 7.0587082, -6.6569500, 7.0587082, -13.7156572, 13.7156572
7: -8.3428059, 4.1933599, -8.3428059, 4.1933599, -12.5361652, 12.5361652
8: -8.1921329, 6.0312138, -8.1921329, 6.0312138, -14.2233467, 14.2233467
9: -6.3373275, 6.7819953, -6.3373275, 6.7819953, -13.1193199, 13.1193218

Time for backsubstitution: 1.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087835, upper bound: 9.0087845
time: 2.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087835, upper bound: 9.0087846
time: 1.93 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6137094, 5.8590560, -7.6137094, 5.8590560, -13.4727650, 13.4727640
1: -6.1292791, 5.3075094, -6.1292791, 5.3075094, -11.4367886, 11.4367886
2: -7.4691033, 4.5115442, -7.4691033, 4.5115442, -11.9806461, 11.9806480
3: -9.0654783, 4.4164257, -9.0654783, 4.4164257, -13.4819021, 13.4819012
4: -8.2032833, 6.3942995, -8.2032833, 6.3942995, -14.5975828, 14.5975819
5: -6.5569053, 5.5308518, -6.5569053, 5.5308518, -12.0877571, 12.0877571
6: -6.6569500, 7.0587082, -6.6569500, 7.0587082, -13.7156572, 13.7156572
7: -8.3428059, 4.1933599, -8.3428059, 4.1933599, -12.5361652, 12.5361652
8: -8.1921329, 6.0312138, -8.1921329, 6.0312138, -14.2233467, 14.2233467
9: -6.3373275, 6.7819953, -6.3373275, 6.7819953, -13.1193199, 13.1193218

Time for backsubstitution: 2.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.27 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087835, upper bound: 9.0087845
time: 2.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087835, upper bound: 9.0087846
time: 1.82 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6137094, 5.8590560, -7.6137094, 5.8590560, -13.4727650, 13.4727640
1: -6.1292791, 5.3075094, -6.1292791, 5.3075094, -11.4367886, 11.4367886
2: -7.4691033, 4.5115442, -7.4691033, 4.5115442, -11.9806461, 11.9806480
3: -9.0654783, 4.4164257, -9.0654783, 4.4164257, -13.4819021, 13.4819012
4: -8.2032833, 6.3942995, -8.2032833, 6.3942995, -14.5975828, 14.5975819
5: -6.5569053, 5.5308518, -6.5569053, 5.5308518, -12.0877571, 12.0877571
6: -6.6569500, 7.0587082, -6.6569500, 7.0587082, -13.7156572, 13.7156572
7: -8.3428059, 4.1933599, -8.3428059, 4.1933599, -12.5361652, 12.5361652
8: -8.1921329, 6.0312138, -8.1921329, 6.0312138, -14.2233467, 14.2233467
9: -6.3373275, 6.7819953, -6.3373275, 6.7819953, -13.1193199, 13.1193218

Time for backsubstitution: 1.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087835
time: 2.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087844, upper bound: 9.0087835
time: 2.00 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6137094, 5.8590560, -7.6137094, 5.8590560, -13.4727650, 13.4727640
1: -6.1292791, 5.3075094, -6.1292791, 5.3075094, -11.4367886, 11.4367886
2: -7.4691033, 4.5115442, -7.4691033, 4.5115442, -11.9806461, 11.9806480
3: -9.0654783, 4.4164257, -9.0654783, 4.4164257, -13.4819021, 13.4819012
4: -8.2032833, 6.3942995, -8.2032833, 6.3942995, -14.5975828, 14.5975819
5: -6.5569053, 5.5308518, -6.5569053, 5.5308518, -12.0877571, 12.0877571
6: -6.6569500, 7.0587082, -6.6569500, 7.0587082, -13.7156572, 13.7156572
7: -8.3428059, 4.1933599, -8.3428059, 4.1933599, -12.5361652, 12.5361652
8: -8.1921329, 6.0312138, -8.1921329, 6.0312138, -14.2233467, 14.2233467
9: -6.3373275, 6.7819953, -6.3373275, 6.7819953, -13.1193199, 13.1193218

Time for backsubstitution: 2.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087835
time: 1.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087845, upper bound: 9.0087835
time: 2.77 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6137094, 5.8590560, -7.6137094, 5.8590560, -13.4727650, 13.4727640
1: -6.1292791, 5.3075094, -6.1292791, 5.3075094, -11.4367886, 11.4367886
2: -7.4691033, 4.5115442, -7.4691033, 4.5115442, -11.9806461, 11.9806480
3: -9.0654783, 4.4164257, -9.0654783, 4.4164257, -13.4819021, 13.4819012
4: -8.2032833, 6.3942995, -8.2032833, 6.3942995, -14.5975828, 14.5975819
5: -6.5569053, 5.5308518, -6.5569053, 5.5308518, -12.0877571, 12.0877571
6: -6.6569500, 7.0587082, -6.6569500, 7.0587082, -13.7156572, 13.7156572
7: -8.3428059, 4.1933599, -8.3428059, 4.1933599, -12.5361652, 12.5361652
8: -8.1921329, 6.0312138, -8.1921329, 6.0312138, -14.2233467, 14.2233467
9: -6.3373275, 6.7819953, -6.3373275, 6.7819953, -13.1193199, 13.1193218

Time for backsubstitution: 2.01 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 7.43 + 594.57 = 602.00 seconds
