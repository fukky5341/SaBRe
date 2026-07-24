## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 9.007671511800002


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

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
execution time: IAR + RelationalAnalysis = 1.44 + 4.94 = 6.38 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -9.0166882, upper bound: 9.0166883

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0095889, upper bound: 9.0095889
time: 3.15 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0095889, upper bound: 9.0095889
time: 3.15 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 6.45 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 6.45
Output dim: 7, lower bound: -9.0095889, upper bound: 9.0095889
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 6.45
Output dim: 7, lower bound: -9.0095889, upper bound: 9.0095889

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0094059, upper bound: 9.0094059
time: 1.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0094060, upper bound: 9.0094060
time: 2.05 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0094059, upper bound: 9.0094059
time: 1.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0094060, upper bound: 9.0094060
time: 2.00 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 7.40 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 7.40
Output dim: 7, lower bound: -9.0094059, upper bound: 9.0094059
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 7.40
Output dim: 7, lower bound: -9.0094060, upper bound: 9.0094060
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 7.40
Output dim: 7, lower bound: -9.0094059, upper bound: 9.0094059
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 7.40
Output dim: 7, lower bound: -9.0094060, upper bound: 9.0094060

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093946, upper bound: 9.0093943
time: 2.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093943, upper bound: 9.0093946
time: 2.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093946, upper bound: 9.0093944
time: 2.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093943, upper bound: 9.0093946
time: 1.89 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093946, upper bound: 9.0093943
time: 2.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093943, upper bound: 9.0093946
time: 2.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093946, upper bound: 9.0093944
time: 2.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093943, upper bound: 9.0093946
time: 1.89 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 8.18 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 8.18
Output dim: 7, lower bound: -9.0093946, upper bound: 9.0093943
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 8.18
Output dim: 7, lower bound: -9.0093943, upper bound: 9.0093946
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 8.18
Output dim: 7, lower bound: -9.0093946, upper bound: 9.0093944
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 8.18
Output dim: 7, lower bound: -9.0093943, upper bound: 9.0093946
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 8.18
Output dim: 7, lower bound: -9.0093946, upper bound: 9.0093943
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 8.18
Output dim: 7, lower bound: -9.0093943, upper bound: 9.0093946
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 8.18
Output dim: 7, lower bound: -9.0093946, upper bound: 9.0093944
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 8.18
Output dim: 7, lower bound: -9.0093943, upper bound: 9.0093946

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093946, upper bound: 9.0093925
time: 1.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093927, upper bound: 9.0093944
time: 2.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093944, upper bound: 9.0093927
time: 2.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093925, upper bound: 9.0093946
time: 8.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093946, upper bound: 9.0093925
time: 1.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093927, upper bound: 9.0093943
time: 2.24 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093944, upper bound: 9.0093927
time: 2.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093925, upper bound: 9.0093946
time: 1.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093946, upper bound: 9.0093925
time: 1.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093927, upper bound: 9.0093944
time: 2.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093944, upper bound: 9.0093927
time: 2.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093925, upper bound: 9.0093946
time: 8.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093946, upper bound: 9.0093925
time: 1.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093927, upper bound: 9.0093943
time: 2.23 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093944, upper bound: 9.0093927
time: 2.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093925, upper bound: 9.0093946
time: 1.70 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 7.81 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 7.81
Output dim: 7, lower bound: -9.0093946, upper bound: 9.0093925
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 7.81
Output dim: 7, lower bound: -9.0093927, upper bound: 9.0093944
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 7.81
Output dim: 7, lower bound: -9.0093944, upper bound: 9.0093927
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 7.81
Output dim: 7, lower bound: -9.0093925, upper bound: 9.0093946
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 7.81
Output dim: 7, lower bound: -9.0093946, upper bound: 9.0093925
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 7.81
Output dim: 7, lower bound: -9.0093927, upper bound: 9.0093943
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 7.81
Output dim: 7, lower bound: -9.0093944, upper bound: 9.0093927
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 7.81
Output dim: 7, lower bound: -9.0093925, upper bound: 9.0093946
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 7.81
Output dim: 7, lower bound: -9.0093946, upper bound: 9.0093925
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 7.81
Output dim: 7, lower bound: -9.0093927, upper bound: 9.0093944
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 7.81
Output dim: 7, lower bound: -9.0093944, upper bound: 9.0093927
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 7.81
Output dim: 7, lower bound: -9.0093925, upper bound: 9.0093946
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 7.81
Output dim: 7, lower bound: -9.0093946, upper bound: 9.0093925
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 7.81
Output dim: 7, lower bound: -9.0093927, upper bound: 9.0093943
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 7.81
Output dim: 7, lower bound: -9.0093944, upper bound: 9.0093927
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 7.81
Output dim: 7, lower bound: -9.0093925, upper bound: 9.0093946

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093946, upper bound: 9.0093925
time: 2.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093946, upper bound: 9.0093925
time: 1.91 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093926, upper bound: 9.0093944
time: 2.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093927, upper bound: 9.0093943
time: 2.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093943, upper bound: 9.0093927
time: 1.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093943, upper bound: 9.0093927
time: 2.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093924, upper bound: 9.0093946
time: 1.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093925, upper bound: 9.0093946
time: 2.12 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093946, upper bound: 9.0093925
time: 3.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093946, upper bound: 9.0093924
time: 2.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093927, upper bound: 9.0093944
time: 2.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093927, upper bound: 9.0093943
time: 2.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093943, upper bound: 9.0093927
time: 1.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093943, upper bound: 9.0093926
time: 1.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093924, upper bound: 9.0093946
time: 1.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093925, upper bound: 9.0093946
time: 1.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093946, upper bound: 9.0093925
time: 2.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093946, upper bound: 9.0093925
time: 1.92 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093926, upper bound: 9.0093944
time: 2.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093927, upper bound: 9.0093943
time: 2.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093943, upper bound: 9.0093927
time: 1.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093943, upper bound: 9.0093927
time: 2.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093924, upper bound: 9.0093946
time: 1.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093925, upper bound: 9.0093946
time: 2.14 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093946, upper bound: 9.0093925
time: 3.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093946, upper bound: 9.0093924
time: 2.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093927, upper bound: 9.0093944
time: 2.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093927, upper bound: 9.0093943
time: 2.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093943, upper bound: 9.0093927
time: 1.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093943, upper bound: 9.0093926
time: 2.00 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093924, upper bound: 9.0093946
time: 1.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093925, upper bound: 9.0093946
time: 1.71 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 6.92 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.92
Output dim: 7, lower bound: -9.0093946, upper bound: 9.0093925
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.92
Output dim: 7, lower bound: -9.0093946, upper bound: 9.0093925
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.92
Output dim: 7, lower bound: -9.0093926, upper bound: 9.0093944
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.92
Output dim: 7, lower bound: -9.0093927, upper bound: 9.0093943
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.92
Output dim: 7, lower bound: -9.0093943, upper bound: 9.0093927
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.92
Output dim: 7, lower bound: -9.0093943, upper bound: 9.0093927
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.92
Output dim: 7, lower bound: -9.0093924, upper bound: 9.0093946
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.92
Output dim: 7, lower bound: -9.0093925, upper bound: 9.0093946
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.92
Output dim: 7, lower bound: -9.0093946, upper bound: 9.0093925
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.92
Output dim: 7, lower bound: -9.0093946, upper bound: 9.0093924
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.92
Output dim: 7, lower bound: -9.0093927, upper bound: 9.0093944
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.92
Output dim: 7, lower bound: -9.0093927, upper bound: 9.0093943
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.92
Output dim: 7, lower bound: -9.0093943, upper bound: 9.0093927
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.92
Output dim: 7, lower bound: -9.0093943, upper bound: 9.0093926
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.92
Output dim: 7, lower bound: -9.0093924, upper bound: 9.0093946
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.92
Output dim: 7, lower bound: -9.0093925, upper bound: 9.0093946
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.92
Output dim: 7, lower bound: -9.0093946, upper bound: 9.0093925
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.92
Output dim: 7, lower bound: -9.0093946, upper bound: 9.0093925
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.92
Output dim: 7, lower bound: -9.0093926, upper bound: 9.0093944
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.92
Output dim: 7, lower bound: -9.0093927, upper bound: 9.0093943
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.92
Output dim: 7, lower bound: -9.0093943, upper bound: 9.0093927
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.92
Output dim: 7, lower bound: -9.0093943, upper bound: 9.0093927
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.92
Output dim: 7, lower bound: -9.0093924, upper bound: 9.0093946
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.92
Output dim: 7, lower bound: -9.0093925, upper bound: 9.0093946
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.92
Output dim: 7, lower bound: -9.0093946, upper bound: 9.0093925
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.92
Output dim: 7, lower bound: -9.0093946, upper bound: 9.0093924
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.92
Output dim: 7, lower bound: -9.0093927, upper bound: 9.0093944
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.92
Output dim: 7, lower bound: -9.0093927, upper bound: 9.0093943
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.92
Output dim: 7, lower bound: -9.0093943, upper bound: 9.0093927
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.92
Output dim: 7, lower bound: -9.0093943, upper bound: 9.0093926
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.92
Output dim: 7, lower bound: -9.0093924, upper bound: 9.0093946
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.92
Output dim: 7, lower bound: -9.0093925, upper bound: 9.0093946

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087835
time: 6.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087836
time: 2.96 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087835
time: 2.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087835
time: 2.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087841, upper bound: 9.0087844
time: 3.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087840, upper bound: 9.0087843
time: 2.18 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087841, upper bound: 9.0087843
time: 1.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087841, upper bound: 9.0087843
time: 2.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087843, upper bound: 9.0087841
time: 2.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087843, upper bound: 9.0087841
time: 1.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087843, upper bound: 9.0087841
time: 1.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087843, upper bound: 9.0087841
time: 2.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087835, upper bound: 9.0087846
time: 2.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087835, upper bound: 9.0087846
time: 1.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087835, upper bound: 9.0087846
time: 1.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087835, upper bound: 9.0087846
time: 2.44 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087835
time: 2.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087835
time: 7.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087835
time: 2.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087835
time: 2.00 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087841, upper bound: 9.0087843
time: 2.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087840, upper bound: 9.0087843
time: 1.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087841, upper bound: 9.0087843
time: 2.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087841, upper bound: 9.0087843
time: 2.46 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087843, upper bound: 9.0087841
time: 3.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087843, upper bound: 9.0087841
time: 1.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087843, upper bound: 9.0087840
time: 1.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087844, upper bound: 9.0087841
time: 2.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087835, upper bound: 9.0087846
time: 2.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087835, upper bound: 9.0087846
time: 1.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087836, upper bound: 9.0087846
time: 3.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087835, upper bound: 9.0087846
time: 2.26 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087835
time: 6.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087836
time: 2.99 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087835
time: 2.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087835
time: 2.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087841, upper bound: 9.0087844
time: 3.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087840, upper bound: 9.0087843
time: 2.16 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087841, upper bound: 9.0087843
time: 1.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087841, upper bound: 9.0087843
time: 2.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087843, upper bound: 9.0087841
time: 2.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087843, upper bound: 9.0087841
time: 1.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087843, upper bound: 9.0087841
time: 1.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087843, upper bound: 9.0087841
time: 2.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087835, upper bound: 9.0087846
time: 2.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087835, upper bound: 9.0087846
time: 1.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087835, upper bound: 9.0087846
time: 1.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087835, upper bound: 9.0087846
time: 2.44 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087835
time: 2.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087835
time: 7.91 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087835
time: 2.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087835
time: 2.00 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087841, upper bound: 9.0087843
time: 2.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087840, upper bound: 9.0087843
time: 1.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087841, upper bound: 9.0087843
time: 2.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087841, upper bound: 9.0087843
time: 2.47 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087843, upper bound: 9.0087841
time: 4.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087843, upper bound: 9.0087841
time: 1.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087843, upper bound: 9.0087840
time: 1.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087844, upper bound: 9.0087841
time: 2.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087835, upper bound: 9.0087846
time: 2.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087835, upper bound: 9.0087846
time: 1.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087836, upper bound: 9.0087846
time: 3.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087835, upper bound: 9.0087846
time: 2.30 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 9.81 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.81
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087835
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.81
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087836
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.81
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087835
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.81
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087835
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.81
Output dim: 7, lower bound: -9.0087841, upper bound: 9.0087844
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.81
Output dim: 7, lower bound: -9.0087840, upper bound: 9.0087843
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.81
Output dim: 7, lower bound: -9.0087841, upper bound: 9.0087843
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.81
Output dim: 7, lower bound: -9.0087841, upper bound: 9.0087843
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.81
Output dim: 7, lower bound: -9.0087843, upper bound: 9.0087841
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.81
Output dim: 7, lower bound: -9.0087843, upper bound: 9.0087841
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.81
Output dim: 7, lower bound: -9.0087843, upper bound: 9.0087841
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.81
Output dim: 7, lower bound: -9.0087843, upper bound: 9.0087841
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.81
Output dim: 7, lower bound: -9.0087835, upper bound: 9.0087846
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.81
Output dim: 7, lower bound: -9.0087835, upper bound: 9.0087846
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.81
Output dim: 7, lower bound: -9.0087835, upper bound: 9.0087846
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.81
Output dim: 7, lower bound: -9.0087835, upper bound: 9.0087846
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.81
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087835
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.81
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087835
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.81
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087835
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.81
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087835
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.81
Output dim: 7, lower bound: -9.0087841, upper bound: 9.0087843
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.81
Output dim: 7, lower bound: -9.0087840, upper bound: 9.0087843
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.81
Output dim: 7, lower bound: -9.0087841, upper bound: 9.0087843
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.81
Output dim: 7, lower bound: -9.0087841, upper bound: 9.0087843
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.81
Output dim: 7, lower bound: -9.0087843, upper bound: 9.0087841
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.81
Output dim: 7, lower bound: -9.0087843, upper bound: 9.0087841
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.81
Output dim: 7, lower bound: -9.0087843, upper bound: 9.0087840
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.81
Output dim: 7, lower bound: -9.0087844, upper bound: 9.0087841
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.81
Output dim: 7, lower bound: -9.0087835, upper bound: 9.0087846
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.81
Output dim: 7, lower bound: -9.0087835, upper bound: 9.0087846
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.81
Output dim: 7, lower bound: -9.0087836, upper bound: 9.0087846
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.81
Output dim: 7, lower bound: -9.0087835, upper bound: 9.0087846
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.81
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087835
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.81
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087836
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.81
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087835
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.81
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087835
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.81
Output dim: 7, lower bound: -9.0087841, upper bound: 9.0087844
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.81
Output dim: 7, lower bound: -9.0087840, upper bound: 9.0087843
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.81
Output dim: 7, lower bound: -9.0087841, upper bound: 9.0087843
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.81
Output dim: 7, lower bound: -9.0087841, upper bound: 9.0087843
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.81
Output dim: 7, lower bound: -9.0087843, upper bound: 9.0087841
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.81
Output dim: 7, lower bound: -9.0087843, upper bound: 9.0087841
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.81
Output dim: 7, lower bound: -9.0087843, upper bound: 9.0087841
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.81
Output dim: 7, lower bound: -9.0087843, upper bound: 9.0087841
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.81
Output dim: 7, lower bound: -9.0087835, upper bound: 9.0087846
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.81
Output dim: 7, lower bound: -9.0087835, upper bound: 9.0087846
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.81
Output dim: 7, lower bound: -9.0087835, upper bound: 9.0087846
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.81
Output dim: 7, lower bound: -9.0087835, upper bound: 9.0087846
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.81
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087835
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.81
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087835
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.81
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087835
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.81
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087835
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.81
Output dim: 7, lower bound: -9.0087841, upper bound: 9.0087843
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.81
Output dim: 7, lower bound: -9.0087840, upper bound: 9.0087843
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.81
Output dim: 7, lower bound: -9.0087841, upper bound: 9.0087843
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.81
Output dim: 7, lower bound: -9.0087841, upper bound: 9.0087843
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.81
Output dim: 7, lower bound: -9.0087843, upper bound: 9.0087841
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.81
Output dim: 7, lower bound: -9.0087843, upper bound: 9.0087841
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.81
Output dim: 7, lower bound: -9.0087843, upper bound: 9.0087840
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.81
Output dim: 7, lower bound: -9.0087844, upper bound: 9.0087841
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.81
Output dim: 7, lower bound: -9.0087835, upper bound: 9.0087846
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.81
Output dim: 7, lower bound: -9.0087835, upper bound: 9.0087846
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.81
Output dim: 7, lower bound: -9.0087836, upper bound: 9.0087846
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.81
Output dim: 7, lower bound: -9.0087835, upper bound: 9.0087846

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087835
time: 2.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087845, upper bound: 9.0087835
time: 1.50 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087835
time: 2.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087845, upper bound: 9.0087835
time: 3.25 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087835
time: 1.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087845, upper bound: 9.0087835
time: 4.72 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087834
time: 2.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087845, upper bound: 9.0087835
time: 1.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087841, upper bound: 9.0087843
time: 1.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087840, upper bound: 9.0087844
time: 1.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087840, upper bound: 9.0087842
time: 1.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087840, upper bound: 9.0087843
time: 2.51 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=51, inp2_unstable=51, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=229, inp2_unstable=229, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087841, upper bound: 9.0087842
time: 1.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0087840, upper bound: 9.0087843
time: 4.28 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 9.64 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 9.64
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087835
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 9.64
Output dim: 7, lower bound: -9.0087845, upper bound: 9.0087835
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 9.64
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087835
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 9.64
Output dim: 7, lower bound: -9.0087845, upper bound: 9.0087835
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 9.64
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087835
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 9.64
Output dim: 7, lower bound: -9.0087845, upper bound: 9.0087835
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 9.64
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087834
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 9.64
Output dim: 7, lower bound: -9.0087845, upper bound: 9.0087835
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 9.64
Output dim: 7, lower bound: -9.0087841, upper bound: 9.0087843
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 9.64
Output dim: 7, lower bound: -9.0087840, upper bound: 9.0087844
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 9.64
Output dim: 7, lower bound: -9.0087840, upper bound: 9.0087842
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 9.64
Output dim: 7, lower bound: -9.0087840, upper bound: 9.0087843
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 9.64
Output dim: 7, lower bound: -9.0087841, upper bound: 9.0087842
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 9.64
Output dim: 7, lower bound: -9.0087840, upper bound: 9.0087843
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.64
Output dim: 7, lower bound: -9.0087841, upper bound: 9.0087843
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.64
Output dim: 7, lower bound: -9.0087843, upper bound: 9.0087841
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.64
Output dim: 7, lower bound: -9.0087843, upper bound: 9.0087841
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.64
Output dim: 7, lower bound: -9.0087843, upper bound: 9.0087841
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.64
Output dim: 7, lower bound: -9.0087843, upper bound: 9.0087841
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.64
Output dim: 7, lower bound: -9.0087835, upper bound: 9.0087846
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.64
Output dim: 7, lower bound: -9.0087835, upper bound: 9.0087846
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.64
Output dim: 7, lower bound: -9.0087835, upper bound: 9.0087846
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.64
Output dim: 7, lower bound: -9.0087835, upper bound: 9.0087846
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.64
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087835
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.64
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087835
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.64
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087835
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.64
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087835
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.64
Output dim: 7, lower bound: -9.0087841, upper bound: 9.0087843
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.64
Output dim: 7, lower bound: -9.0087840, upper bound: 9.0087843
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.64
Output dim: 7, lower bound: -9.0087841, upper bound: 9.0087843
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.64
Output dim: 7, lower bound: -9.0087841, upper bound: 9.0087843
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.64
Output dim: 7, lower bound: -9.0087843, upper bound: 9.0087841
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.64
Output dim: 7, lower bound: -9.0087843, upper bound: 9.0087841
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.64
Output dim: 7, lower bound: -9.0087843, upper bound: 9.0087840
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.64
Output dim: 7, lower bound: -9.0087844, upper bound: 9.0087841
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.64
Output dim: 7, lower bound: -9.0087835, upper bound: 9.0087846
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.64
Output dim: 7, lower bound: -9.0087835, upper bound: 9.0087846
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.64
Output dim: 7, lower bound: -9.0087836, upper bound: 9.0087846
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.64
Output dim: 7, lower bound: -9.0087835, upper bound: 9.0087846
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.64
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087835
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.64
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087836
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.64
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087835
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.64
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087835
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.64
Output dim: 7, lower bound: -9.0087841, upper bound: 9.0087844
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.64
Output dim: 7, lower bound: -9.0087840, upper bound: 9.0087843
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.64
Output dim: 7, lower bound: -9.0087841, upper bound: 9.0087843
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.64
Output dim: 7, lower bound: -9.0087841, upper bound: 9.0087843
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.64
Output dim: 7, lower bound: -9.0087843, upper bound: 9.0087841
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.64
Output dim: 7, lower bound: -9.0087843, upper bound: 9.0087841
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.64
Output dim: 7, lower bound: -9.0087843, upper bound: 9.0087841
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.64
Output dim: 7, lower bound: -9.0087843, upper bound: 9.0087841
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.64
Output dim: 7, lower bound: -9.0087835, upper bound: 9.0087846
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.64
Output dim: 7, lower bound: -9.0087835, upper bound: 9.0087846
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.64
Output dim: 7, lower bound: -9.0087835, upper bound: 9.0087846
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.64
Output dim: 7, lower bound: -9.0087835, upper bound: 9.0087846
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.64
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087835
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.64
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087835
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.64
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087835
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.64
Output dim: 7, lower bound: -9.0087846, upper bound: 9.0087835
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.64
Output dim: 7, lower bound: -9.0087841, upper bound: 9.0087843
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.64
Output dim: 7, lower bound: -9.0087840, upper bound: 9.0087843
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.64
Output dim: 7, lower bound: -9.0087841, upper bound: 9.0087843
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.64
Output dim: 7, lower bound: -9.0087841, upper bound: 9.0087843
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.64
Output dim: 7, lower bound: -9.0087843, upper bound: 9.0087841
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.64
Output dim: 7, lower bound: -9.0087843, upper bound: 9.0087841
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.64
Output dim: 7, lower bound: -9.0087843, upper bound: 9.0087840
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.64
Output dim: 7, lower bound: -9.0087844, upper bound: 9.0087841
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.64
Output dim: 7, lower bound: -9.0087835, upper bound: 9.0087846
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.64
Output dim: 7, lower bound: -9.0087835, upper bound: 9.0087846
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.64
Output dim: 7, lower bound: -9.0087836, upper bound: 9.0087846
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.64
Output dim: 7, lower bound: -9.0087835, upper bound: 9.0087846

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 6.38 + 594.91 = 601.29 seconds
