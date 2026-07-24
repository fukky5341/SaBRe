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
execution time: IAR + RelationalAnalysis = 0.89 + 4.75 = 5.64 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -9.0166882, upper bound: 9.0166883

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 72

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0149741, upper bound: 9.0149741
time: 3.52 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0149741, upper bound: 9.0149741
time: 3.49 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 7.03 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 7.03
Output dim: 7, lower bound: -9.0149741, upper bound: 9.0149741
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 7.03
Output dim: 7, lower bound: -9.0149741, upper bound: 9.0149741

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 57

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0147819, upper bound: 9.0147819
time: 6.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0147819, upper bound: 9.0147819
time: 2.44 seconds

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 153

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 210

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0142724, upper bound: 9.0142726
time: 2.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0142724, upper bound: 9.0142724
time: 2.56 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 7.81 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 7.81
Output dim: 7, lower bound: -9.0147819, upper bound: 9.0147819
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 7.81
Output dim: 7, lower bound: -9.0147819, upper bound: 9.0147819
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 7.81
Output dim: 7, lower bound: -9.0142724, upper bound: 9.0142726
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 7.81
Output dim: 7, lower bound: -9.0142724, upper bound: 9.0142724

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 127

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0130776, upper bound: 9.0130776
time: 3.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0130776, upper bound: 9.0130776
time: 3.16 seconds

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 97

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 61

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 54

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0133283, upper bound: 9.0133281
time: 2.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0133283, upper bound: 9.0133281
time: 2.02 seconds

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0132866, upper bound: 9.0132866
time: 2.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0132866, upper bound: 9.0132866
time: 2.28 seconds

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 232

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 219

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0128405, upper bound: 9.0128405
time: 6.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0128405, upper bound: 9.0128405
time: 2.23 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 11.05 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 11.05
Output dim: 7, lower bound: -9.0130776, upper bound: 9.0130776
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 11.05
Output dim: 7, lower bound: -9.0130776, upper bound: 9.0130776
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 11.05
Output dim: 7, lower bound: -9.0133283, upper bound: 9.0133281
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 11.05
Output dim: 7, lower bound: -9.0133283, upper bound: 9.0133281
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 11.05
Output dim: 7, lower bound: -9.0132866, upper bound: 9.0132866
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 11.05
Output dim: 7, lower bound: -9.0132866, upper bound: 9.0132866
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 11.05
Output dim: 7, lower bound: -9.0128405, upper bound: 9.0128405
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 11.05
Output dim: 7, lower bound: -9.0128405, upper bound: 9.0128405

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 232

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 210

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0122885, upper bound: 9.0122892
time: 2.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0122885, upper bound: 9.0122892
time: 2.14 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0125949, upper bound: 9.0125915
time: 2.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0125916, upper bound: 9.0125949
time: 3.01 seconds

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 113

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0133145, upper bound: 9.0133162
time: 2.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0133168, upper bound: 9.0133142
time: 2.09 seconds

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 85

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0131383, upper bound: 9.0131383
time: 3.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0131383, upper bound: 9.0131383
time: 10.07 seconds

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 138

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 242

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0131683, upper bound: 9.0131684
time: 2.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0131684, upper bound: 9.0131683
time: 3.41 seconds

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0132866, upper bound: 9.0132855
time: 2.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0132855, upper bound: 9.0132866
time: 3.77 seconds

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 50

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0128204, upper bound: 9.0128223
time: 1.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0128223, upper bound: 9.0128204
time: 8.99 seconds

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 84

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 61

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0103412, upper bound: 9.0103412
time: 1.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0103412, upper bound: 9.0103412
time: 1.94 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 6.54 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 6.54
Output dim: 7, lower bound: -9.0122885, upper bound: 9.0122892
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 6.54
Output dim: 7, lower bound: -9.0122885, upper bound: 9.0122892
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 6.54
Output dim: 7, lower bound: -9.0125949, upper bound: 9.0125915
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 6.54
Output dim: 7, lower bound: -9.0125916, upper bound: 9.0125949
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 6.54
Output dim: 7, lower bound: -9.0133145, upper bound: 9.0133162
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 6.54
Output dim: 7, lower bound: -9.0133168, upper bound: 9.0133142
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 6.54
Output dim: 7, lower bound: -9.0131383, upper bound: 9.0131383
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 6.54
Output dim: 7, lower bound: -9.0131383, upper bound: 9.0131383
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 6.54
Output dim: 7, lower bound: -9.0131683, upper bound: 9.0131684
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 6.54
Output dim: 7, lower bound: -9.0131684, upper bound: 9.0131683
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 6.54
Output dim: 7, lower bound: -9.0132866, upper bound: 9.0132855
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 6.54
Output dim: 7, lower bound: -9.0132855, upper bound: 9.0132866
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 6.54
Output dim: 7, lower bound: -9.0128204, upper bound: 9.0128223
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 6.54
Output dim: 7, lower bound: -9.0128223, upper bound: 9.0128204
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 6.54
Output dim: 7, lower bound: -9.0103412, upper bound: 9.0103412
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 6.54
Output dim: 7, lower bound: -9.0103412, upper bound: 9.0103412

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 138

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 61

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0106645, upper bound: 9.0106644
time: 2.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0106645, upper bound: 9.0106644
time: 2.28 seconds

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 97

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0121804, upper bound: 9.0121862
time: 2.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0121856, upper bound: 9.0121810
time: 2.10 seconds

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 138

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 113

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0125783, upper bound: 9.0125791
time: 1.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0125823, upper bound: 9.0125730
time: 2.85 seconds

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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 138

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0125916, upper bound: 9.0125908
time: 2.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0125859, upper bound: 9.0125949
time: 1.70 seconds

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0133145, upper bound: 9.0133137
time: 1.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0133127, upper bound: 9.0133162
time: 3.12 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 85

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0131180, upper bound: 9.0131126
time: 2.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0131180, upper bound: 9.0131126
time: 2.64 seconds

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 138

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0126931, upper bound: 9.0126930
time: 3.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0126931, upper bound: 9.0126930
time: 2.03 seconds

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 50

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 210

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0120510, upper bound: 9.0120512
time: 2.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0120510, upper bound: 9.0120512
time: 2.04 seconds

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 127

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0107298, upper bound: 9.0107297
time: 1.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0107298, upper bound: 9.0107297
time: 1.60 seconds

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0129923, upper bound: 9.0129925
time: 3.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0129927, upper bound: 9.0129923
time: 3.42 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0132745, upper bound: 9.0132778
time: 3.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0132790, upper bound: 9.0132723
time: 1.97 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 84

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0132723, upper bound: 9.0132790
time: 4.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0132778, upper bound: 9.0132745
time: 3.14 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 127

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0105754, upper bound: 9.0105769
time: 4.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0105754, upper bound: 9.0105769
time: 4.01 seconds

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0103256, upper bound: 9.0103224
time: 1.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0103256, upper bound: 9.0103224
time: 1.90 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 113

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 68

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0103412, upper bound: 9.0103405
time: 2.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0103404, upper bound: 9.0103412
time: 1.90 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 232

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 57

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0101706, upper bound: 9.0101706
time: 2.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0101706, upper bound: 9.0101706
time: 4.36 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 7.45 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 7.45
Output dim: 7, lower bound: -9.0106645, upper bound: 9.0106644
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 7.45
Output dim: 7, lower bound: -9.0106645, upper bound: 9.0106644
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 7.45
Output dim: 7, lower bound: -9.0121804, upper bound: 9.0121862
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 7.45
Output dim: 7, lower bound: -9.0121856, upper bound: 9.0121810
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 7.45
Output dim: 7, lower bound: -9.0125783, upper bound: 9.0125791
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 7.45
Output dim: 7, lower bound: -9.0125823, upper bound: 9.0125730
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 7.45
Output dim: 7, lower bound: -9.0125916, upper bound: 9.0125908
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 7.45
Output dim: 7, lower bound: -9.0125859, upper bound: 9.0125949
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 7.45
Output dim: 7, lower bound: -9.0133145, upper bound: 9.0133137
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 7.45
Output dim: 7, lower bound: -9.0133127, upper bound: 9.0133162
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 7.45
Output dim: 7, lower bound: -9.0131180, upper bound: 9.0131126
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 7.45
Output dim: 7, lower bound: -9.0131180, upper bound: 9.0131126
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 7.45
Output dim: 7, lower bound: -9.0126931, upper bound: 9.0126930
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 7.45
Output dim: 7, lower bound: -9.0126931, upper bound: 9.0126930
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 7.45
Output dim: 7, lower bound: -9.0120510, upper bound: 9.0120512
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 7.45
Output dim: 7, lower bound: -9.0120510, upper bound: 9.0120512
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 7.45
Output dim: 7, lower bound: -9.0107298, upper bound: 9.0107297
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 7.45
Output dim: 7, lower bound: -9.0107298, upper bound: 9.0107297
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 7.45
Output dim: 7, lower bound: -9.0129923, upper bound: 9.0129925
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 7.45
Output dim: 7, lower bound: -9.0129927, upper bound: 9.0129923
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 7.45
Output dim: 7, lower bound: -9.0132745, upper bound: 9.0132778
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 7.45
Output dim: 7, lower bound: -9.0132790, upper bound: 9.0132723
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 7.45
Output dim: 7, lower bound: -9.0132723, upper bound: 9.0132790
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 7.45
Output dim: 7, lower bound: -9.0132778, upper bound: 9.0132745
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 7.45
Output dim: 7, lower bound: -9.0105754, upper bound: 9.0105769
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 7.45
Output dim: 7, lower bound: -9.0105754, upper bound: 9.0105769
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 7.45
Output dim: 7, lower bound: -9.0103256, upper bound: 9.0103224
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 7.45
Output dim: 7, lower bound: -9.0103256, upper bound: 9.0103224
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 7.45
Output dim: 7, lower bound: -9.0103412, upper bound: 9.0103405
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 7.45
Output dim: 7, lower bound: -9.0103404, upper bound: 9.0103412
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 7.45
Output dim: 7, lower bound: -9.0101706, upper bound: 9.0101706
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 7.45
Output dim: 7, lower bound: -9.0101706, upper bound: 9.0101706

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 68

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0106645, upper bound: 9.0106643
time: 1.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0106643, upper bound: 9.0106644
time: 3.23 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 50

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0106426, upper bound: 9.0106443
time: 2.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0106443, upper bound: 9.0106426
time: 3.30 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 50

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 85

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0117838, upper bound: 9.0117872
time: 2.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0117838, upper bound: 9.0117872
time: 2.48 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 97

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0121810, upper bound: 9.0121810
time: 3.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0121856, upper bound: 9.0121782
time: 1.82 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0125290, upper bound: 9.0125341
time: 5.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0125322, upper bound: 9.0125301
time: 3.24 seconds

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 153

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0125531, upper bound: 9.0125418
time: 5.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0125510, upper bound: 9.0125448
time: 3.29 seconds

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0115803, upper bound: 9.0115820
time: 4.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0115811, upper bound: 9.0115807
time: 2.19 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 232

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 97

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0125832, upper bound: 9.0125949
time: 7.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0125859, upper bound: 9.0125899
time: 2.91 seconds

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0117007, upper bound: 9.0117022
time: 2.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0117007, upper bound: 9.0117022
time: 2.22 seconds

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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 127

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0111868, upper bound: 9.0111900
time: 2.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0111868, upper bound: 9.0111900
time: 2.51 seconds

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

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0115107, upper bound: 9.0115091
time: 2.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0115107, upper bound: 9.0115091
time: 2.79 seconds

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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0115107, upper bound: 9.0115091
time: 2.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0115107, upper bound: 9.0115091
time: 2.41 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 232

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 50

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0126689, upper bound: 9.0126794
time: 2.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0126795, upper bound: 9.0126687
time: 4.44 seconds

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

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 68

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0126931, upper bound: 9.0126926
time: 2.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0126926, upper bound: 9.0126930
time: 6.53 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 61

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0120510, upper bound: 9.0120470
time: 1.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0120469, upper bound: 9.0120512
time: 2.26 seconds

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

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 50

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0120319, upper bound: 9.0120349
time: 1.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0120347, upper bound: 9.0120322
time: 2.08 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 97

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 219

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 153

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0107051, upper bound: 9.0107013
time: 7.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0107019, upper bound: 9.0107050
time: 1.71 seconds

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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0107298, upper bound: 9.0107296
time: 2.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0107298, upper bound: 9.0107297
time: 1.99 seconds

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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0109237, upper bound: 9.0109229
time: 2.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0109237, upper bound: 9.0109229
time: 2.14 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 138

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0129927, upper bound: 9.0129865
time: 4.05 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0129872, upper bound: 9.0129923
time: 3.58 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 50

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 232

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 57

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0130961, upper bound: 9.0131003
time: 3.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0130965, upper bound: 9.0130976
time: 2.58 seconds

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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 219

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0113692, upper bound: 9.0113669
time: 2.02 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0113692, upper bound: 9.0113669
time: 2.03 seconds

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

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0124545, upper bound: 9.0124619
time: 3.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0124551, upper bound: 9.0124611
time: 6.21 seconds

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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0132778, upper bound: 9.0132745
time: 2.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0132778, upper bound: 9.0132745
time: 2.18 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0105250, upper bound: 9.0105267
time: 2.00 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0105252, upper bound: 9.0105267
time: 1.69 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 84

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 97

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0105741, upper bound: 9.0105769
time: 3.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0105754, upper bound: 9.0105752
time: 5.28 seconds

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

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 54

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 134

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 85

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0100520, upper bound: 9.0100491
time: 2.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0100520, upper bound: 9.0100491
time: 2.74 seconds

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

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 68

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0103256, upper bound: 9.0103222
time: 2.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0103248, upper bound: 9.0103224
time: 1.93 seconds

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

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 127

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 138

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0099919, upper bound: 9.0099908
time: 1.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0099919, upper bound: 9.0099908
time: 1.61 seconds

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

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 84

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0103112, upper bound: 9.0103131
time: 4.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0103124, upper bound: 9.0103126
time: 1.67 seconds

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

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 54

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0100379, upper bound: 9.0100386
time: 2.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0100386, upper bound: 9.0100380
time: 1.78 seconds

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

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 97

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0101697, upper bound: 9.0101706
time: 2.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0101706, upper bound: 9.0101697
time: 2.75 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 6.24 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -9.0106645, upper bound: 9.0106643
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -9.0106643, upper bound: 9.0106644
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -9.0106426, upper bound: 9.0106443
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -9.0106443, upper bound: 9.0106426
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -9.0117838, upper bound: 9.0117872
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -9.0117838, upper bound: 9.0117872
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -9.0121810, upper bound: 9.0121810
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -9.0121856, upper bound: 9.0121782
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -9.0125290, upper bound: 9.0125341
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -9.0125322, upper bound: 9.0125301
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -9.0125531, upper bound: 9.0125418
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -9.0125510, upper bound: 9.0125448
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -9.0115803, upper bound: 9.0115820
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -9.0115811, upper bound: 9.0115807
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -9.0125832, upper bound: 9.0125949
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -9.0125859, upper bound: 9.0125899
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -9.0117007, upper bound: 9.0117022
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -9.0117007, upper bound: 9.0117022
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -9.0111868, upper bound: 9.0111900
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -9.0111868, upper bound: 9.0111900
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -9.0115107, upper bound: 9.0115091
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -9.0115107, upper bound: 9.0115091
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -9.0115107, upper bound: 9.0115091
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -9.0115107, upper bound: 9.0115091
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -9.0126689, upper bound: 9.0126794
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -9.0126795, upper bound: 9.0126687
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -9.0126931, upper bound: 9.0126926
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -9.0126926, upper bound: 9.0126930
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -9.0120510, upper bound: 9.0120470
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -9.0120469, upper bound: 9.0120512
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -9.0120319, upper bound: 9.0120349
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -9.0120347, upper bound: 9.0120322
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -9.0107051, upper bound: 9.0107013
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -9.0107019, upper bound: 9.0107050
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -9.0107298, upper bound: 9.0107296
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -9.0107298, upper bound: 9.0107297
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -9.0109237, upper bound: 9.0109229
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -9.0109237, upper bound: 9.0109229
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -9.0129927, upper bound: 9.0129865
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -9.0129872, upper bound: 9.0129923
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -9.0130961, upper bound: 9.0131003
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -9.0130965, upper bound: 9.0130976
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -9.0113692, upper bound: 9.0113669
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -9.0113692, upper bound: 9.0113669
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -9.0124545, upper bound: 9.0124619
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -9.0124551, upper bound: 9.0124611
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -9.0132778, upper bound: 9.0132745
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -9.0132778, upper bound: 9.0132745
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -9.0105250, upper bound: 9.0105267
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -9.0105252, upper bound: 9.0105267
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -9.0105741, upper bound: 9.0105769
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -9.0105754, upper bound: 9.0105752
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -9.0100520, upper bound: 9.0100491
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -9.0100520, upper bound: 9.0100491
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -9.0103256, upper bound: 9.0103222
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -9.0103248, upper bound: 9.0103224
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -9.0099919, upper bound: 9.0099908
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -9.0099919, upper bound: 9.0099908
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -9.0103112, upper bound: 9.0103131
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -9.0103124, upper bound: 9.0103126
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -9.0100379, upper bound: 9.0100386
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -9.0100386, upper bound: 9.0100380
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -9.0101697, upper bound: 9.0101706
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -9.0101706, upper bound: 9.0101697

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 153

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0106378, upper bound: 9.0106340
time: 1.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0106344, upper bound: 9.0106376
time: 3.44 seconds

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 113

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0106576, upper bound: 9.0106584
time: 2.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0106584, upper bound: 9.0106575
time: 3.01 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 97

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0105146, upper bound: 9.0105170
time: 2.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0105143, upper bound: 9.0105170
time: 2.16 seconds

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

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 84

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 85

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0103008, upper bound: 9.0103005
time: 4.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0103008, upper bound: 9.0103005
time: 5.30 seconds

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

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 97

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 153

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0117593, upper bound: 9.0117616
time: 3.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0117587, upper bound: 9.0117623
time: 4.08 seconds

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

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0102110, upper bound: 9.0102115
time: 15.12 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0102110, upper bound: 9.0102115
time: 20.51 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 84

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 68

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0121810, upper bound: 9.0121797
time: 3.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0121808, upper bound: 9.0121810
time: 5.71 seconds

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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 219

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0102795, upper bound: 9.0102776
time: 2.09 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0102795, upper bound: 9.0102776
time: 2.11 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 232

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 54

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0105542, upper bound: 9.0105591
time: 1.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0105542, upper bound: 9.0105591
time: 1.92 seconds

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

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0125322, upper bound: 9.0125294
time: 2.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0125319, upper bound: 9.0125301
time: 3.15 seconds

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

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 138

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0110409, upper bound: 9.0110291
time: 3.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0110409, upper bound: 9.0110290
time: 3.16 seconds

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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 85

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0122312, upper bound: 9.0122247
time: 5.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0122312, upper bound: 9.0122247
time: 6.65 seconds

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

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 232

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0115261, upper bound: 9.0115316
time: 2.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0115295, upper bound: 9.0115269
time: 2.82 seconds

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

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 54

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0114310, upper bound: 9.0114341
time: 2.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0114345, upper bound: 9.0114275
time: 7.39 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 10.62 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 10.62
Output dim: 7, lower bound: -9.0106378, upper bound: 9.0106340
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 10.62
Output dim: 7, lower bound: -9.0106344, upper bound: 9.0106376
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 10.62
Output dim: 7, lower bound: -9.0106576, upper bound: 9.0106584
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 10.62
Output dim: 7, lower bound: -9.0106584, upper bound: 9.0106575
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 10.62
Output dim: 7, lower bound: -9.0105146, upper bound: 9.0105170
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 10.62
Output dim: 7, lower bound: -9.0105143, upper bound: 9.0105170
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 10.62
Output dim: 7, lower bound: -9.0103008, upper bound: 9.0103005
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 10.62
Output dim: 7, lower bound: -9.0103008, upper bound: 9.0103005
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 10.62
Output dim: 7, lower bound: -9.0117593, upper bound: 9.0117616
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 10.62
Output dim: 7, lower bound: -9.0117587, upper bound: 9.0117623
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 10.62
Output dim: 7, lower bound: -9.0102110, upper bound: 9.0102115
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 10.62
Output dim: 7, lower bound: -9.0102110, upper bound: 9.0102115
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 10.62
Output dim: 7, lower bound: -9.0121810, upper bound: 9.0121797
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 10.62
Output dim: 7, lower bound: -9.0121808, upper bound: 9.0121810
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 10.62
Output dim: 7, lower bound: -9.0102795, upper bound: 9.0102776
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 10.62
Output dim: 7, lower bound: -9.0102795, upper bound: 9.0102776
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 10.62
Output dim: 7, lower bound: -9.0105542, upper bound: 9.0105591
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 10.62
Output dim: 7, lower bound: -9.0105542, upper bound: 9.0105591
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 10.62
Output dim: 7, lower bound: -9.0125322, upper bound: 9.0125294
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 10.62
Output dim: 7, lower bound: -9.0125319, upper bound: 9.0125301
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 10.62
Output dim: 7, lower bound: -9.0110409, upper bound: 9.0110291
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 10.62
Output dim: 7, lower bound: -9.0110409, upper bound: 9.0110290
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 10.62
Output dim: 7, lower bound: -9.0122312, upper bound: 9.0122247
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 10.62
Output dim: 7, lower bound: -9.0122312, upper bound: 9.0122247
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 10.62
Output dim: 7, lower bound: -9.0115261, upper bound: 9.0115316
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 10.62
Output dim: 7, lower bound: -9.0115295, upper bound: 9.0115269
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 10.62
Output dim: 7, lower bound: -9.0114310, upper bound: 9.0114341
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 10.62
Output dim: 7, lower bound: -9.0114345, upper bound: 9.0114275
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 10.62
Output dim: 7, lower bound: -9.0125832, upper bound: 9.0125949
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 10.62
Output dim: 7, lower bound: -9.0125859, upper bound: 9.0125899
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 10.62
Output dim: 7, lower bound: -9.0117007, upper bound: 9.0117022
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 10.62
Output dim: 7, lower bound: -9.0117007, upper bound: 9.0117022
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 10.62
Output dim: 7, lower bound: -9.0111868, upper bound: 9.0111900
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 10.62
Output dim: 7, lower bound: -9.0111868, upper bound: 9.0111900
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 10.62
Output dim: 7, lower bound: -9.0115107, upper bound: 9.0115091
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 10.62
Output dim: 7, lower bound: -9.0115107, upper bound: 9.0115091
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 10.62
Output dim: 7, lower bound: -9.0115107, upper bound: 9.0115091
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 10.62
Output dim: 7, lower bound: -9.0115107, upper bound: 9.0115091
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 10.62
Output dim: 7, lower bound: -9.0126689, upper bound: 9.0126794
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 10.62
Output dim: 7, lower bound: -9.0126795, upper bound: 9.0126687
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 10.62
Output dim: 7, lower bound: -9.0126931, upper bound: 9.0126926
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 10.62
Output dim: 7, lower bound: -9.0126926, upper bound: 9.0126930
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 10.62
Output dim: 7, lower bound: -9.0120510, upper bound: 9.0120470
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 10.62
Output dim: 7, lower bound: -9.0120469, upper bound: 9.0120512
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 10.62
Output dim: 7, lower bound: -9.0120319, upper bound: 9.0120349
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 10.62
Output dim: 7, lower bound: -9.0120347, upper bound: 9.0120322
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 10.62
Output dim: 7, lower bound: -9.0107051, upper bound: 9.0107013
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 10.62
Output dim: 7, lower bound: -9.0107019, upper bound: 9.0107050
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 10.62
Output dim: 7, lower bound: -9.0107298, upper bound: 9.0107296
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 10.62
Output dim: 7, lower bound: -9.0107298, upper bound: 9.0107297
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 10.62
Output dim: 7, lower bound: -9.0109237, upper bound: 9.0109229
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 10.62
Output dim: 7, lower bound: -9.0109237, upper bound: 9.0109229
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 10.62
Output dim: 7, lower bound: -9.0129927, upper bound: 9.0129865
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 10.62
Output dim: 7, lower bound: -9.0129872, upper bound: 9.0129923
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 10.62
Output dim: 7, lower bound: -9.0130961, upper bound: 9.0131003
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 10.62
Output dim: 7, lower bound: -9.0130965, upper bound: 9.0130976
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 10.62
Output dim: 7, lower bound: -9.0113692, upper bound: 9.0113669
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 10.62
Output dim: 7, lower bound: -9.0113692, upper bound: 9.0113669
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 10.62
Output dim: 7, lower bound: -9.0124545, upper bound: 9.0124619
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 10.62
Output dim: 7, lower bound: -9.0124551, upper bound: 9.0124611
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 10.62
Output dim: 7, lower bound: -9.0132778, upper bound: 9.0132745
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 10.62
Output dim: 7, lower bound: -9.0132778, upper bound: 9.0132745
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 10.62
Output dim: 7, lower bound: -9.0105250, upper bound: 9.0105267
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 10.62
Output dim: 7, lower bound: -9.0105252, upper bound: 9.0105267
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 10.62
Output dim: 7, lower bound: -9.0105741, upper bound: 9.0105769
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 10.62
Output dim: 7, lower bound: -9.0105754, upper bound: 9.0105752
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 10.62
Output dim: 7, lower bound: -9.0100520, upper bound: 9.0100491
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 10.62
Output dim: 7, lower bound: -9.0100520, upper bound: 9.0100491
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 10.62
Output dim: 7, lower bound: -9.0103256, upper bound: 9.0103222
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 10.62
Output dim: 7, lower bound: -9.0103248, upper bound: 9.0103224
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 10.62
Output dim: 7, lower bound: -9.0099919, upper bound: 9.0099908
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 10.62
Output dim: 7, lower bound: -9.0099919, upper bound: 9.0099908
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 10.62
Output dim: 7, lower bound: -9.0103112, upper bound: 9.0103131
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 10.62
Output dim: 7, lower bound: -9.0103124, upper bound: 9.0103126
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 10.62
Output dim: 7, lower bound: -9.0100379, upper bound: 9.0100386
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 10.62
Output dim: 7, lower bound: -9.0100386, upper bound: 9.0100380
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 10.62
Output dim: 7, lower bound: -9.0101697, upper bound: 9.0101706
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 10.62
Output dim: 7, lower bound: -9.0101706, upper bound: 9.0101697

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 5.64 + 601.59 = 607.23 seconds
