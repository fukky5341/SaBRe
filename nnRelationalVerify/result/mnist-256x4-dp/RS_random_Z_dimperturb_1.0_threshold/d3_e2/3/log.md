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
execution time: IAR + RelationalAnalysis = 1.36 + 4.98 = 6.33 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -9.0166882, upper bound: 9.0166883

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 138

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0157411, upper bound: 9.0157411
time: 2.17 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0157411, upper bound: 9.0157411
time: 2.20 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 4.38 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 4.38
Output dim: 7, lower bound: -9.0157411, upper bound: 9.0157411
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 4.38
Output dim: 7, lower bound: -9.0157411, upper bound: 9.0157411

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

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0152399, upper bound: 9.0152405
time: 2.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0152405, upper bound: 9.0152399
time: 2.59 seconds

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

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0155563, upper bound: 9.0155563
time: 2.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0155563, upper bound: 9.0155563
time: 2.74 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 6.75 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 6.75
Output dim: 7, lower bound: -9.0152399, upper bound: 9.0152405
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 6.75
Output dim: 7, lower bound: -9.0152405, upper bound: 9.0152399
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 6.75
Output dim: 7, lower bound: -9.0155563, upper bound: 9.0155563
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 6.75
Output dim: 7, lower bound: -9.0155563, upper bound: 9.0155563

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

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0140551, upper bound: 9.0140557
time: 3.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0140551, upper bound: 9.0140557
time: 3.16 seconds

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

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 97

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0152396, upper bound: 9.0152399
time: 2.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0152405, upper bound: 9.0152391
time: 3.42 seconds

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

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 54

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0143874, upper bound: 9.0143874
time: 3.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0143874, upper bound: 9.0143874
time: 3.47 seconds

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

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 113

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 232

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0088785, upper bound: 9.0088785
time: 1.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0088785, upper bound: 9.0088785
time: 1.73 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 4.71 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.71
Output dim: 7, lower bound: -9.0140551, upper bound: 9.0140557
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.71
Output dim: 7, lower bound: -9.0140551, upper bound: 9.0140557
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.71
Output dim: 7, lower bound: -9.0152396, upper bound: 9.0152399
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.71
Output dim: 7, lower bound: -9.0152405, upper bound: 9.0152391
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.71
Output dim: 7, lower bound: -9.0143874, upper bound: 9.0143874
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.71
Output dim: 7, lower bound: -9.0143874, upper bound: 9.0143874
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.71
Output dim: 7, lower bound: -9.0088785, upper bound: 9.0088785
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.71
Output dim: 7, lower bound: -9.0088785, upper bound: 9.0088785

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

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0137679, upper bound: 9.0137680
time: 3.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0137679, upper bound: 9.0137680
time: 2.74 seconds

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

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 54

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0124759, upper bound: 9.0124759
time: 1.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0124759, upper bound: 9.0124759
time: 1.86 seconds

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

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0137853, upper bound: 9.0137860
time: 2.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0137853, upper bound: 9.0137860
time: 2.06 seconds

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

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 153

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0132757, upper bound: 9.0132752
time: 5.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0132757, upper bound: 9.0132752
time: 5.64 seconds

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

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0139969, upper bound: 9.0139961
time: 3.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0139961, upper bound: 9.0139969
time: 2.84 seconds

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

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0143874, upper bound: 9.0143862
time: 4.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0143862, upper bound: 9.0143874
time: 3.66 seconds

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

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0088785, upper bound: 9.0088782
time: 2.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0088782, upper bound: 9.0088785
time: 1.76 seconds

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

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0086658, upper bound: 9.0086663
time: 1.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0086663, upper bound: 9.0086658
time: 2.24 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 5.31 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.31
Output dim: 7, lower bound: -9.0137679, upper bound: 9.0137680
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.31
Output dim: 7, lower bound: -9.0137679, upper bound: 9.0137680
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.31
Output dim: 7, lower bound: -9.0124759, upper bound: 9.0124759
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.31
Output dim: 7, lower bound: -9.0124759, upper bound: 9.0124759
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.31
Output dim: 7, lower bound: -9.0137853, upper bound: 9.0137860
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.31
Output dim: 7, lower bound: -9.0137853, upper bound: 9.0137860
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.31
Output dim: 7, lower bound: -9.0132757, upper bound: 9.0132752
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.31
Output dim: 7, lower bound: -9.0132757, upper bound: 9.0132752
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.31
Output dim: 7, lower bound: -9.0139969, upper bound: 9.0139961
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.31
Output dim: 7, lower bound: -9.0139961, upper bound: 9.0139969
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.31
Output dim: 7, lower bound: -9.0143874, upper bound: 9.0143862
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.31
Output dim: 7, lower bound: -9.0143862, upper bound: 9.0143874
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.31
Output dim: 7, lower bound: -9.0088785, upper bound: 9.0088782
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.31
Output dim: 7, lower bound: -9.0088782, upper bound: 9.0088785
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.31
Output dim: 7, lower bound: -9.0086658, upper bound: 9.0086663
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.31
Output dim: 7, lower bound: -9.0086663, upper bound: 9.0086658

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

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0137679, upper bound: 9.0137654
time: 2.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0137648, upper bound: 9.0137680
time: 2.62 seconds

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

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 153

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 57

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0135725, upper bound: 9.0135719
time: 2.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0135719, upper bound: 9.0135726
time: 3.36 seconds

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

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0124759, upper bound: 9.0124751
time: 2.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0124751, upper bound: 9.0124759
time: 1.90 seconds

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

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 232

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0124471, upper bound: 9.0124487
time: 2.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0124487, upper bound: 9.0124471
time: 6.30 seconds

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

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 232

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0119049, upper bound: 9.0119072
time: 2.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0119049, upper bound: 9.0119072
time: 2.12 seconds

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

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0134257, upper bound: 9.0134279
time: 2.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0134257, upper bound: 9.0134279
time: 3.15 seconds

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

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0129596, upper bound: 9.0129591
time: 3.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0129592, upper bound: 9.0129597
time: 2.23 seconds

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

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 50

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0129996, upper bound: 9.0129999
time: 2.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0130001, upper bound: 9.0129994
time: 2.53 seconds

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

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 57

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0137765, upper bound: 9.0137760
time: 3.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0137767, upper bound: 9.0137758
time: 3.23 seconds

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

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 113

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0139699, upper bound: 9.0139784
time: 3.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0139777, upper bound: 9.0139703
time: 3.62 seconds

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

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0143866, upper bound: 9.0143821
time: 5.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0143834, upper bound: 9.0143856
time: 2.70 seconds

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

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0143689, upper bound: 9.0143699
time: 2.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0143692, upper bound: 9.0143692
time: 3.19 seconds

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

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 153

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0081911, upper bound: 9.0081908
time: 2.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0081911, upper bound: 9.0081909
time: 3.35 seconds

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

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0088782, upper bound: 9.0088781
time: 2.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0088778, upper bound: 9.0088785
time: 2.21 seconds

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

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 84

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 97

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0086658, upper bound: 9.0086663
time: 3.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0086658, upper bound: 9.0086661
time: 1.40 seconds

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

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 84

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 57

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0084288, upper bound: 9.0084278
time: 1.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0084288, upper bound: 9.0084280
time: 2.00 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 12.86 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.86
Output dim: 7, lower bound: -9.0137679, upper bound: 9.0137654
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.86
Output dim: 7, lower bound: -9.0137648, upper bound: 9.0137680
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.86
Output dim: 7, lower bound: -9.0135725, upper bound: 9.0135719
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.86
Output dim: 7, lower bound: -9.0135719, upper bound: 9.0135726
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.86
Output dim: 7, lower bound: -9.0124759, upper bound: 9.0124751
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.86
Output dim: 7, lower bound: -9.0124751, upper bound: 9.0124759
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.86
Output dim: 7, lower bound: -9.0124471, upper bound: 9.0124487
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.86
Output dim: 7, lower bound: -9.0124487, upper bound: 9.0124471
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.86
Output dim: 7, lower bound: -9.0119049, upper bound: 9.0119072
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.86
Output dim: 7, lower bound: -9.0119049, upper bound: 9.0119072
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.86
Output dim: 7, lower bound: -9.0134257, upper bound: 9.0134279
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.86
Output dim: 7, lower bound: -9.0134257, upper bound: 9.0134279
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.86
Output dim: 7, lower bound: -9.0129596, upper bound: 9.0129591
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.86
Output dim: 7, lower bound: -9.0129592, upper bound: 9.0129597
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.86
Output dim: 7, lower bound: -9.0129996, upper bound: 9.0129999
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.86
Output dim: 7, lower bound: -9.0130001, upper bound: 9.0129994
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.86
Output dim: 7, lower bound: -9.0137765, upper bound: 9.0137760
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.86
Output dim: 7, lower bound: -9.0137767, upper bound: 9.0137758
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.86
Output dim: 7, lower bound: -9.0139699, upper bound: 9.0139784
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.86
Output dim: 7, lower bound: -9.0139777, upper bound: 9.0139703
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.86
Output dim: 7, lower bound: -9.0143866, upper bound: 9.0143821
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.86
Output dim: 7, lower bound: -9.0143834, upper bound: 9.0143856
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.86
Output dim: 7, lower bound: -9.0143689, upper bound: 9.0143699
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.86
Output dim: 7, lower bound: -9.0143692, upper bound: 9.0143692
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.86
Output dim: 7, lower bound: -9.0081911, upper bound: 9.0081908
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.86
Output dim: 7, lower bound: -9.0081911, upper bound: 9.0081909
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.86
Output dim: 7, lower bound: -9.0088782, upper bound: 9.0088781
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.86
Output dim: 7, lower bound: -9.0088778, upper bound: 9.0088785
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.86
Output dim: 7, lower bound: -9.0086658, upper bound: 9.0086663
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.86
Output dim: 7, lower bound: -9.0086658, upper bound: 9.0086661
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.86
Output dim: 7, lower bound: -9.0084288, upper bound: 9.0084278
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.86
Output dim: 7, lower bound: -9.0084288, upper bound: 9.0084280

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

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0137673, upper bound: 9.0137654
time: 2.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0137679, upper bound: 9.0137651
time: 2.08 seconds

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

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 84

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0115978, upper bound: 9.0115997
time: 2.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0115978, upper bound: 9.0115997
time: 2.30 seconds

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

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0135720, upper bound: 9.0135719
time: 4.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0135725, upper bound: 9.0135713
time: 2.58 seconds

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

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 97

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0135699, upper bound: 9.0135726
time: 4.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0135719, upper bound: 9.0135710
time: 5.93 seconds

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

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0124756, upper bound: 9.0124751
time: 1.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0124759, upper bound: 9.0124750
time: 3.12 seconds

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

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0124751, upper bound: 9.0124749
time: 2.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0124742, upper bound: 9.0124759
time: 14.33 seconds

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

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 72

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0123573, upper bound: 9.0123622
time: 2.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0123611, upper bound: 9.0123592
time: 7.25 seconds

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

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0120608, upper bound: 9.0120591
time: 2.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0120604, upper bound: 9.0120593
time: 2.12 seconds

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

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 54

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0118832, upper bound: 9.0118850
time: 10.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0118829, upper bound: 9.0118856
time: 3.74 seconds

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

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093800, upper bound: 9.0093808
time: 2.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0093800, upper bound: 9.0093808
time: 2.07 seconds

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

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0134080, upper bound: 9.0134088
time: 2.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0134068, upper bound: 9.0134100
time: 3.47 seconds

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

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 54

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0116611, upper bound: 9.0116623
time: 2.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0116611, upper bound: 9.0116623
time: 2.94 seconds

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

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 138

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0129596, upper bound: 9.0129576
time: 2.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0129582, upper bound: 9.0129591
time: 3.91 seconds

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

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0126759, upper bound: 9.0126767
time: 3.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0126759, upper bound: 9.0126767
time: 2.55 seconds

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

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0129620, upper bound: 9.0129601
time: 2.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0129604, upper bound: 9.0129613
time: 2.41 seconds

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

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0129845, upper bound: 9.0129878
time: 3.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0129885, upper bound: 9.0129844
time: 6.16 seconds

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

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0119982, upper bound: 9.0119977
time: 3.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0119982, upper bound: 9.0119977
time: 3.85 seconds

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

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0117218, upper bound: 9.0117223
time: 7.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0117218, upper bound: 9.0117223
time: 6.54 seconds

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

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0121758, upper bound: 9.0121813
time: 3.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0121758, upper bound: 9.0121813
time: 3.73 seconds

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

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0129440, upper bound: 9.0129373
time: 3.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0129440, upper bound: 9.0129373
time: 2.59 seconds

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

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0125242, upper bound: 9.0125211
time: 3.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0125242, upper bound: 9.0125211
time: 3.14 seconds

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

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0136164, upper bound: 9.0136182
time: 6.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0136169, upper bound: 9.0136181
time: 6.90 seconds

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

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0141927, upper bound: 9.0141971
time: 3.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0141957, upper bound: 9.0141944
time: 3.69 seconds

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

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 84

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0143688, upper bound: 9.0143658
time: 3.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0143652, upper bound: 9.0143690
time: 2.38 seconds

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

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0081861, upper bound: 9.0081850
time: 2.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0081855, upper bound: 9.0081856
time: 1.59 seconds

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

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 57

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0079573, upper bound: 9.0079570
time: 2.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0079571, upper bound: 9.0079573
time: 2.92 seconds

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

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 50

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 97

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0088781, upper bound: 9.0088781
time: 1.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0088782, upper bound: 9.0088781
time: 1.88 seconds

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

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 113

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0088774, upper bound: 9.0088785
time: 2.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0088778, upper bound: 9.0088784
time: 1.55 seconds

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

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 84

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0086658, upper bound: 9.0086661
time: 2.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0086652, upper bound: 9.0086663
time: 5.75 seconds

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

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 138

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0086658, upper bound: 9.0086660
time: 1.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0086658, upper bound: 9.0086661
time: 1.87 seconds

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

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0084288, upper bound: 9.0084272
time: 2.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0084285, upper bound: 9.0084278
time: 1.90 seconds

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

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 54

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0081080, upper bound: 9.0081073
time: 3.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0081080, upper bound: 9.0081073
time: 2.09 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 6.42 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.42
Output dim: 7, lower bound: -9.0137673, upper bound: 9.0137654
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.42
Output dim: 7, lower bound: -9.0137679, upper bound: 9.0137651
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.42
Output dim: 7, lower bound: -9.0115978, upper bound: 9.0115997
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.42
Output dim: 7, lower bound: -9.0115978, upper bound: 9.0115997
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.42
Output dim: 7, lower bound: -9.0135720, upper bound: 9.0135719
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.42
Output dim: 7, lower bound: -9.0135725, upper bound: 9.0135713
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.42
Output dim: 7, lower bound: -9.0135699, upper bound: 9.0135726
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.42
Output dim: 7, lower bound: -9.0135719, upper bound: 9.0135710
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.42
Output dim: 7, lower bound: -9.0124756, upper bound: 9.0124751
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.42
Output dim: 7, lower bound: -9.0124759, upper bound: 9.0124750
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.42
Output dim: 7, lower bound: -9.0124751, upper bound: 9.0124749
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.42
Output dim: 7, lower bound: -9.0124742, upper bound: 9.0124759
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.42
Output dim: 7, lower bound: -9.0123573, upper bound: 9.0123622
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.42
Output dim: 7, lower bound: -9.0123611, upper bound: 9.0123592
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.42
Output dim: 7, lower bound: -9.0120608, upper bound: 9.0120591
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.42
Output dim: 7, lower bound: -9.0120604, upper bound: 9.0120593
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.42
Output dim: 7, lower bound: -9.0118832, upper bound: 9.0118850
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.42
Output dim: 7, lower bound: -9.0118829, upper bound: 9.0118856
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.42
Output dim: 7, lower bound: -9.0093800, upper bound: 9.0093808
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.42
Output dim: 7, lower bound: -9.0093800, upper bound: 9.0093808
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.42
Output dim: 7, lower bound: -9.0134080, upper bound: 9.0134088
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.42
Output dim: 7, lower bound: -9.0134068, upper bound: 9.0134100
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.42
Output dim: 7, lower bound: -9.0116611, upper bound: 9.0116623
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.42
Output dim: 7, lower bound: -9.0116611, upper bound: 9.0116623
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.42
Output dim: 7, lower bound: -9.0129596, upper bound: 9.0129576
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.42
Output dim: 7, lower bound: -9.0129582, upper bound: 9.0129591
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.42
Output dim: 7, lower bound: -9.0126759, upper bound: 9.0126767
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.42
Output dim: 7, lower bound: -9.0126759, upper bound: 9.0126767
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.42
Output dim: 7, lower bound: -9.0129620, upper bound: 9.0129601
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.42
Output dim: 7, lower bound: -9.0129604, upper bound: 9.0129613
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.42
Output dim: 7, lower bound: -9.0129845, upper bound: 9.0129878
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.42
Output dim: 7, lower bound: -9.0129885, upper bound: 9.0129844
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.42
Output dim: 7, lower bound: -9.0119982, upper bound: 9.0119977
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.42
Output dim: 7, lower bound: -9.0119982, upper bound: 9.0119977
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.42
Output dim: 7, lower bound: -9.0117218, upper bound: 9.0117223
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.42
Output dim: 7, lower bound: -9.0117218, upper bound: 9.0117223
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.42
Output dim: 7, lower bound: -9.0121758, upper bound: 9.0121813
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.42
Output dim: 7, lower bound: -9.0121758, upper bound: 9.0121813
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.42
Output dim: 7, lower bound: -9.0129440, upper bound: 9.0129373
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.42
Output dim: 7, lower bound: -9.0129440, upper bound: 9.0129373
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.42
Output dim: 7, lower bound: -9.0125242, upper bound: 9.0125211
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.42
Output dim: 7, lower bound: -9.0125242, upper bound: 9.0125211
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.42
Output dim: 7, lower bound: -9.0136164, upper bound: 9.0136182
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.42
Output dim: 7, lower bound: -9.0136169, upper bound: 9.0136181
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.42
Output dim: 7, lower bound: -9.0141927, upper bound: 9.0141971
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.42
Output dim: 7, lower bound: -9.0141957, upper bound: 9.0141944
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.42
Output dim: 7, lower bound: -9.0143688, upper bound: 9.0143658
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.42
Output dim: 7, lower bound: -9.0143652, upper bound: 9.0143690
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.42
Output dim: 7, lower bound: -9.0081861, upper bound: 9.0081850
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.42
Output dim: 7, lower bound: -9.0081855, upper bound: 9.0081856
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.42
Output dim: 7, lower bound: -9.0079573, upper bound: 9.0079570
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.42
Output dim: 7, lower bound: -9.0079571, upper bound: 9.0079573
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.42
Output dim: 7, lower bound: -9.0088781, upper bound: 9.0088781
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.42
Output dim: 7, lower bound: -9.0088782, upper bound: 9.0088781
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.42
Output dim: 7, lower bound: -9.0088774, upper bound: 9.0088785
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.42
Output dim: 7, lower bound: -9.0088778, upper bound: 9.0088784
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.42
Output dim: 7, lower bound: -9.0086658, upper bound: 9.0086661
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.42
Output dim: 7, lower bound: -9.0086652, upper bound: 9.0086663
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.42
Output dim: 7, lower bound: -9.0086658, upper bound: 9.0086660
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.42
Output dim: 7, lower bound: -9.0086658, upper bound: 9.0086661
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.42
Output dim: 7, lower bound: -9.0084288, upper bound: 9.0084272
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.42
Output dim: 7, lower bound: -9.0084285, upper bound: 9.0084278
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.42
Output dim: 7, lower bound: -9.0081080, upper bound: 9.0081073
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.42
Output dim: 7, lower bound: -9.0081080, upper bound: 9.0081073

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

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 54

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0137460, upper bound: 9.0137464
time: 6.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0137483, upper bound: 9.0137442
time: 4.00 seconds

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

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0124109, upper bound: 9.0124105
time: 2.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0124109, upper bound: 9.0124105
time: 2.21 seconds

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

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0115979, upper bound: 9.0115991
time: 2.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0115976, upper bound: 9.0115998
time: 2.73 seconds

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

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 232

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0111443, upper bound: 9.0111449
time: 2.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0111442, upper bound: 9.0111450
time: 2.83 seconds

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

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0135720, upper bound: 9.0135714
time: 3.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0135712, upper bound: 9.0135719
time: 2.45 seconds

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

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 153

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0134048, upper bound: 9.0134087
time: 4.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0134106, upper bound: 9.0134046
time: 3.57 seconds

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

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 84

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0135403, upper bound: 9.0135408
time: 2.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0135369, upper bound: 9.0135441
time: 2.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

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

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 113

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0135480, upper bound: 9.0135500
time: 6.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0135510, upper bound: 9.0135475
time: 3.12 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

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

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 113

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0124541, upper bound: 9.0124549
time: 5.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0124554, upper bound: 9.0124538
time: 2.26 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 13.42 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 13.42
Output dim: 7, lower bound: -9.0137460, upper bound: 9.0137464
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 13.42
Output dim: 7, lower bound: -9.0137483, upper bound: 9.0137442
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 13.42
Output dim: 7, lower bound: -9.0124109, upper bound: 9.0124105
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 13.42
Output dim: 7, lower bound: -9.0124109, upper bound: 9.0124105
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 13.42
Output dim: 7, lower bound: -9.0115979, upper bound: 9.0115991
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 13.42
Output dim: 7, lower bound: -9.0115976, upper bound: 9.0115998
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 13.42
Output dim: 7, lower bound: -9.0111443, upper bound: 9.0111449
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 13.42
Output dim: 7, lower bound: -9.0111442, upper bound: 9.0111450
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 13.42
Output dim: 7, lower bound: -9.0135720, upper bound: 9.0135714
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 13.42
Output dim: 7, lower bound: -9.0135712, upper bound: 9.0135719
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 13.42
Output dim: 7, lower bound: -9.0134048, upper bound: 9.0134087
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 13.42
Output dim: 7, lower bound: -9.0134106, upper bound: 9.0134046
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 13.42
Output dim: 7, lower bound: -9.0135403, upper bound: 9.0135408
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 13.42
Output dim: 7, lower bound: -9.0135369, upper bound: 9.0135441
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 13.42
Output dim: 7, lower bound: -9.0135480, upper bound: 9.0135500
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 13.42
Output dim: 7, lower bound: -9.0135510, upper bound: 9.0135475
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 13.42
Output dim: 7, lower bound: -9.0124541, upper bound: 9.0124549
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 13.42
Output dim: 7, lower bound: -9.0124554, upper bound: 9.0124538
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.42
Output dim: 7, lower bound: -9.0124759, upper bound: 9.0124750
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.42
Output dim: 7, lower bound: -9.0124751, upper bound: 9.0124749
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.42
Output dim: 7, lower bound: -9.0124742, upper bound: 9.0124759
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.42
Output dim: 7, lower bound: -9.0123573, upper bound: 9.0123622
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.42
Output dim: 7, lower bound: -9.0123611, upper bound: 9.0123592
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.42
Output dim: 7, lower bound: -9.0120608, upper bound: 9.0120591
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.42
Output dim: 7, lower bound: -9.0120604, upper bound: 9.0120593
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.42
Output dim: 7, lower bound: -9.0118832, upper bound: 9.0118850
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.42
Output dim: 7, lower bound: -9.0118829, upper bound: 9.0118856
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.42
Output dim: 7, lower bound: -9.0093800, upper bound: 9.0093808
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.42
Output dim: 7, lower bound: -9.0093800, upper bound: 9.0093808
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.42
Output dim: 7, lower bound: -9.0134080, upper bound: 9.0134088
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.42
Output dim: 7, lower bound: -9.0134068, upper bound: 9.0134100
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.42
Output dim: 7, lower bound: -9.0116611, upper bound: 9.0116623
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.42
Output dim: 7, lower bound: -9.0116611, upper bound: 9.0116623
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.42
Output dim: 7, lower bound: -9.0129596, upper bound: 9.0129576
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.42
Output dim: 7, lower bound: -9.0129582, upper bound: 9.0129591
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.42
Output dim: 7, lower bound: -9.0126759, upper bound: 9.0126767
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.42
Output dim: 7, lower bound: -9.0126759, upper bound: 9.0126767
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.42
Output dim: 7, lower bound: -9.0129620, upper bound: 9.0129601
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.42
Output dim: 7, lower bound: -9.0129604, upper bound: 9.0129613
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.42
Output dim: 7, lower bound: -9.0129845, upper bound: 9.0129878
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.42
Output dim: 7, lower bound: -9.0129885, upper bound: 9.0129844
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.42
Output dim: 7, lower bound: -9.0119982, upper bound: 9.0119977
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.42
Output dim: 7, lower bound: -9.0119982, upper bound: 9.0119977
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.42
Output dim: 7, lower bound: -9.0117218, upper bound: 9.0117223
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.42
Output dim: 7, lower bound: -9.0117218, upper bound: 9.0117223
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.42
Output dim: 7, lower bound: -9.0121758, upper bound: 9.0121813
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.42
Output dim: 7, lower bound: -9.0121758, upper bound: 9.0121813
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.42
Output dim: 7, lower bound: -9.0129440, upper bound: 9.0129373
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.42
Output dim: 7, lower bound: -9.0129440, upper bound: 9.0129373
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.42
Output dim: 7, lower bound: -9.0125242, upper bound: 9.0125211
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.42
Output dim: 7, lower bound: -9.0125242, upper bound: 9.0125211
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.42
Output dim: 7, lower bound: -9.0136164, upper bound: 9.0136182
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.42
Output dim: 7, lower bound: -9.0136169, upper bound: 9.0136181
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.42
Output dim: 7, lower bound: -9.0141927, upper bound: 9.0141971
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.42
Output dim: 7, lower bound: -9.0141957, upper bound: 9.0141944
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.42
Output dim: 7, lower bound: -9.0143688, upper bound: 9.0143658
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.42
Output dim: 7, lower bound: -9.0143652, upper bound: 9.0143690
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.42
Output dim: 7, lower bound: -9.0081861, upper bound: 9.0081850
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.42
Output dim: 7, lower bound: -9.0081855, upper bound: 9.0081856
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.42
Output dim: 7, lower bound: -9.0079573, upper bound: 9.0079570
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.42
Output dim: 7, lower bound: -9.0079571, upper bound: 9.0079573
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.42
Output dim: 7, lower bound: -9.0088781, upper bound: 9.0088781
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.42
Output dim: 7, lower bound: -9.0088782, upper bound: 9.0088781
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.42
Output dim: 7, lower bound: -9.0088774, upper bound: 9.0088785
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.42
Output dim: 7, lower bound: -9.0088778, upper bound: 9.0088784
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.42
Output dim: 7, lower bound: -9.0086658, upper bound: 9.0086661
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.42
Output dim: 7, lower bound: -9.0086652, upper bound: 9.0086663
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.42
Output dim: 7, lower bound: -9.0086658, upper bound: 9.0086660
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.42
Output dim: 7, lower bound: -9.0086658, upper bound: 9.0086661
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.42
Output dim: 7, lower bound: -9.0084288, upper bound: 9.0084272
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.42
Output dim: 7, lower bound: -9.0084285, upper bound: 9.0084278
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.42
Output dim: 7, lower bound: -9.0081080, upper bound: 9.0081073
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.42
Output dim: 7, lower bound: -9.0081080, upper bound: 9.0081073

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 6.33 + 599.36 = 605.69 seconds
