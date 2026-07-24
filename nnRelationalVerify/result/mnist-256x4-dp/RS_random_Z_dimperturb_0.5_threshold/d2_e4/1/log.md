## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00710256


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0037834, 0.0086199, 0.0037834, 0.0086199, -0.0048365, 0.0048365)
1: (-0.0002308, 0.0047780, -0.0002308, 0.0047780, -0.0050088, 0.0050088)
2: (-0.0257978, -0.0053192, -0.0257978, -0.0053192, -0.0204786, 0.0204786)
3: (-0.0020375, 0.0080109, -0.0020375, 0.0080109, -0.0100484, 0.0100484)
4: (0.0114507, 0.0185106, 0.0114507, 0.0185106, -0.0070599, 0.0070599)
5: (-0.0032830, 0.0100317, -0.0032830, 0.0100317, -0.0133147, 0.0133147)
6: (0.9943414, 1.0042117, 0.9943414, 1.0042117, -0.0098702, 0.0098702)
7: (0.0073448, 0.0201244, 0.0073448, 0.0201244, -0.0095971, 0.0095971)
8: (0.0018158, 0.0072932, 0.0018158, 0.0072932, -0.0054774, 0.0054774)
9: (-0.0276300, -0.0138944, -0.0276300, -0.0138944, -0.0137356, 0.0137356)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.50 + 2.39 = 3.89 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.0086912, upper bound: 0.0086912

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073653, upper bound: 0.0073653
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073653, upper bound: 0.0073653
time: 1.19 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 2.43 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 2.43
Output dim: 6, lower bound: -0.0073653, upper bound: 0.0073653
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 2.43
Output dim: 6, lower bound: -0.0073653, upper bound: 0.0073653

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037834, 0.0086199, 0.0037834, 0.0086199, -0.0048365, 0.0048365
1: -0.0002308, 0.0047780, -0.0002308, 0.0047780, -0.0050088, 0.0050088
2: -0.0257978, -0.0053192, -0.0257978, -0.0053192, -0.0204786, 0.0204786
3: -0.0020375, 0.0080109, -0.0020375, 0.0080109, -0.0100484, 0.0100484
4: 0.0114507, 0.0185106, 0.0114507, 0.0185106, -0.0070599, 0.0070599
5: -0.0032830, 0.0100317, -0.0032830, 0.0100317, -0.0133147, 0.0133147
6: 0.9943414, 1.0042117, 0.9943414, 1.0042117, -0.0098702, 0.0098702
7: 0.0073448, 0.0201244, 0.0073448, 0.0201244, -0.0095889, 0.0095887
8: 0.0018158, 0.0072932, 0.0018158, 0.0072932, -0.0054774, 0.0054774
9: -0.0276300, -0.0138944, -0.0276300, -0.0138944, -0.0137356, 0.0137356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 153

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073537, upper bound: 0.0073528
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073529, upper bound: 0.0073537
time: 1.12 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037834, 0.0086199, 0.0037834, 0.0086199, -0.0048365, 0.0048365
1: -0.0002308, 0.0047780, -0.0002308, 0.0047780, -0.0050088, 0.0050088
2: -0.0257978, -0.0053192, -0.0257978, -0.0053192, -0.0204786, 0.0204786
3: -0.0020375, 0.0080109, -0.0020375, 0.0080109, -0.0100484, 0.0100484
4: 0.0114507, 0.0185106, 0.0114507, 0.0185106, -0.0070599, 0.0070599
5: -0.0032830, 0.0100317, -0.0032830, 0.0100317, -0.0133147, 0.0133147
6: 0.9943414, 1.0042117, 0.9943414, 1.0042117, -0.0098702, 0.0098702
7: 0.0073448, 0.0201244, 0.0073448, 0.0201244, -0.0095971, 0.0095889
8: 0.0018158, 0.0072932, 0.0018158, 0.0072932, -0.0054774, 0.0054774
9: -0.0276300, -0.0138944, -0.0276300, -0.0138944, -0.0137356, 0.0137356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073537, upper bound: 0.0073529
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073529, upper bound: 0.0073537
time: 1.11 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.61 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.61
Output dim: 6, lower bound: -0.0073537, upper bound: 0.0073528
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.61
Output dim: 6, lower bound: -0.0073529, upper bound: 0.0073537
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.61
Output dim: 6, lower bound: -0.0073537, upper bound: 0.0073529
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.61
Output dim: 6, lower bound: -0.0073529, upper bound: 0.0073537

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037834, 0.0086199, 0.0037834, 0.0086199, -0.0048365, 0.0048365
1: -0.0002308, 0.0047780, -0.0002308, 0.0047780, -0.0050088, 0.0050088
2: -0.0257978, -0.0053192, -0.0257978, -0.0053192, -0.0204786, 0.0204786
3: -0.0020375, 0.0080109, -0.0020375, 0.0080109, -0.0100484, 0.0100484
4: 0.0114507, 0.0185106, 0.0114507, 0.0185106, -0.0070599, 0.0070599
5: -0.0032830, 0.0100317, -0.0032830, 0.0100317, -0.0133147, 0.0133147
6: 0.9943414, 1.0042117, 0.9943414, 1.0042117, -0.0098702, 0.0098702
7: 0.0073448, 0.0201244, 0.0073448, 0.0201244, -0.0095930, 0.0095911
8: 0.0018158, 0.0072932, 0.0018158, 0.0072932, -0.0054774, 0.0054774
9: -0.0276300, -0.0138944, -0.0276300, -0.0138944, -0.0137356, 0.0137356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0068193, upper bound: 0.0068173
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0068193, upper bound: 0.0068173
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037834, 0.0086199, 0.0037834, 0.0086199, -0.0048365, 0.0048365
1: -0.0002308, 0.0047780, -0.0002308, 0.0047780, -0.0050088, 0.0050088
2: -0.0257978, -0.0053192, -0.0257978, -0.0053192, -0.0204786, 0.0204786
3: -0.0020375, 0.0080109, -0.0020375, 0.0080109, -0.0100484, 0.0100484
4: 0.0114507, 0.0185106, 0.0114507, 0.0185106, -0.0070599, 0.0070599
5: -0.0032830, 0.0100317, -0.0032830, 0.0100317, -0.0133147, 0.0133147
6: 0.9943414, 1.0042117, 0.9943414, 1.0042117, -0.0098702, 0.0098702
7: 0.0073448, 0.0201244, 0.0073448, 0.0201244, -0.0095913, 0.0095928
8: 0.0018158, 0.0072932, 0.0018158, 0.0072932, -0.0054774, 0.0054774
9: -0.0276300, -0.0138944, -0.0276300, -0.0138944, -0.0137356, 0.0137356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 165

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073106, upper bound: 0.0072719
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0072712, upper bound: 0.0073114
time: 0.99 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037834, 0.0086199, 0.0037834, 0.0086199, -0.0048365, 0.0048365
1: -0.0002308, 0.0047780, -0.0002308, 0.0047780, -0.0050088, 0.0050088
2: -0.0257978, -0.0053192, -0.0257978, -0.0053192, -0.0204786, 0.0204786
3: -0.0020375, 0.0080109, -0.0020375, 0.0080109, -0.0100484, 0.0100484
4: 0.0114507, 0.0185106, 0.0114507, 0.0185106, -0.0070599, 0.0070599
5: -0.0032830, 0.0100317, -0.0032830, 0.0100317, -0.0133147, 0.0133147
6: 0.9943414, 1.0042117, 0.9943414, 1.0042117, -0.0098702, 0.0098702
7: 0.0073448, 0.0201244, 0.0073448, 0.0201244, -0.0096012, 0.0095913
8: 0.0018158, 0.0072932, 0.0018158, 0.0072932, -0.0054774, 0.0054774
9: -0.0276300, -0.0138944, -0.0276300, -0.0138944, -0.0137356, 0.0137356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0068193, upper bound: 0.0068173
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0068193, upper bound: 0.0068173
time: 1.03 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037834, 0.0086199, 0.0037834, 0.0086199, -0.0048365, 0.0048365
1: -0.0002308, 0.0047780, -0.0002308, 0.0047780, -0.0050088, 0.0050088
2: -0.0257978, -0.0053192, -0.0257978, -0.0053192, -0.0204786, 0.0204786
3: -0.0020375, 0.0080109, -0.0020375, 0.0080109, -0.0100484, 0.0100484
4: 0.0114507, 0.0185106, 0.0114507, 0.0185106, -0.0070599, 0.0070599
5: -0.0032830, 0.0100317, -0.0032830, 0.0100317, -0.0133147, 0.0133147
6: 0.9943414, 1.0042117, 0.9943414, 1.0042117, -0.0098702, 0.0098702
7: 0.0073448, 0.0201244, 0.0073448, 0.0201244, -0.0095995, 0.0095930
8: 0.0018158, 0.0072932, 0.0018158, 0.0072932, -0.0054774, 0.0054774
9: -0.0276300, -0.0138944, -0.0276300, -0.0138944, -0.0137356, 0.0137356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0072650, upper bound: 0.0072601
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0072601, upper bound: 0.0072657
time: 1.16 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.67 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.67
Output dim: 6, lower bound: -0.0068193, upper bound: 0.0068173
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.67
Output dim: 6, lower bound: -0.0068193, upper bound: 0.0068173
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.67
Output dim: 6, lower bound: -0.0073106, upper bound: 0.0072719
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.67
Output dim: 6, lower bound: -0.0072712, upper bound: 0.0073114
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.67
Output dim: 6, lower bound: -0.0068193, upper bound: 0.0068173
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.67
Output dim: 6, lower bound: -0.0068193, upper bound: 0.0068173
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.67
Output dim: 6, lower bound: -0.0072650, upper bound: 0.0072601
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.67
Output dim: 6, lower bound: -0.0072601, upper bound: 0.0072657

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037834, 0.0086199, 0.0037834, 0.0086199, -0.0048365, 0.0048365
1: -0.0002308, 0.0047780, -0.0002308, 0.0047780, -0.0050088, 0.0050088
2: -0.0257978, -0.0053192, -0.0257978, -0.0053192, -0.0204786, 0.0204786
3: -0.0020375, 0.0080109, -0.0020375, 0.0080109, -0.0100484, 0.0100484
4: 0.0114507, 0.0185106, 0.0114507, 0.0185106, -0.0070599, 0.0070599
5: -0.0032830, 0.0100317, -0.0032830, 0.0100317, -0.0133147, 0.0133147
6: 0.9943414, 1.0042117, 0.9943414, 1.0042117, -0.0098702, 0.0098702
7: 0.0073448, 0.0201244, 0.0073448, 0.0201244, -0.0093122, 0.0092606
8: 0.0018158, 0.0072932, 0.0018158, 0.0072932, -0.0054774, 0.0054774
9: -0.0276300, -0.0138944, -0.0276300, -0.0138944, -0.0137356, 0.0137356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073086, upper bound: 0.0072680
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073069, upper bound: 0.0072700
time: 1.12 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037834, 0.0086199, 0.0037834, 0.0086199, -0.0048365, 0.0048365
1: -0.0002308, 0.0047780, -0.0002308, 0.0047780, -0.0050088, 0.0050088
2: -0.0257978, -0.0053192, -0.0257978, -0.0053192, -0.0204786, 0.0204786
3: -0.0020375, 0.0080109, -0.0020375, 0.0080109, -0.0100484, 0.0100484
4: 0.0114507, 0.0185106, 0.0114507, 0.0185106, -0.0070599, 0.0070599
5: -0.0032830, 0.0100317, -0.0032830, 0.0100317, -0.0133147, 0.0133147
6: 0.9943414, 1.0042117, 0.9943414, 1.0042117, -0.0098702, 0.0098702
7: 0.0073448, 0.0201244, 0.0073448, 0.0201244, -0.0092592, 0.0093132
8: 0.0018158, 0.0072932, 0.0018158, 0.0072932, -0.0054774, 0.0054774
9: -0.0276300, -0.0138944, -0.0276300, -0.0138944, -0.0137356, 0.0137356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 75

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 188

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0072545, upper bound: 0.0072894
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0072520, upper bound: 0.0072945
time: 1.22 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037834, 0.0086199, 0.0037834, 0.0086199, -0.0048365, 0.0048365
1: -0.0002308, 0.0047780, -0.0002308, 0.0047780, -0.0050088, 0.0050088
2: -0.0257978, -0.0053192, -0.0257978, -0.0053192, -0.0204786, 0.0204786
3: -0.0020375, 0.0080109, -0.0020375, 0.0080109, -0.0100484, 0.0100484
4: 0.0114507, 0.0185106, 0.0114507, 0.0185106, -0.0070599, 0.0070599
5: -0.0032830, 0.0100317, -0.0032830, 0.0100317, -0.0133147, 0.0133147
6: 0.9943414, 1.0042117, 0.9943414, 1.0042117, -0.0098702, 0.0098702
7: 0.0073448, 0.0201244, 0.0073448, 0.0201244, -0.0095023, 0.0094803
8: 0.0018158, 0.0072932, 0.0018158, 0.0072932, -0.0054774, 0.0054774
9: -0.0276300, -0.0138944, -0.0276300, -0.0138944, -0.0137356, 0.0137356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0060562, upper bound: 0.0060550
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0060562, upper bound: 0.0060550
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037834, 0.0086199, 0.0037834, 0.0086199, -0.0048365, 0.0048365
1: -0.0002308, 0.0047780, -0.0002308, 0.0047780, -0.0050088, 0.0050088
2: -0.0257978, -0.0053192, -0.0257978, -0.0053192, -0.0204786, 0.0204786
3: -0.0020375, 0.0080109, -0.0020375, 0.0080109, -0.0100484, 0.0100484
4: 0.0114507, 0.0185106, 0.0114507, 0.0185106, -0.0070599, 0.0070599
5: -0.0032830, 0.0100317, -0.0032830, 0.0100317, -0.0133147, 0.0133147
6: 0.9943414, 1.0042117, 0.9943414, 1.0042117, -0.0098702, 0.0098702
7: 0.0073448, 0.0201244, 0.0073448, 0.0201244, -0.0094869, 0.0094963
8: 0.0018158, 0.0072932, 0.0018158, 0.0072932, -0.0054774, 0.0054774
9: -0.0276300, -0.0138944, -0.0276300, -0.0138944, -0.0137356, 0.0137356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0072129, upper bound: 0.0072144
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0072129, upper bound: 0.0072144
time: 1.11 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.62 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.62
Output dim: 6, lower bound: -0.0073086, upper bound: 0.0072680
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.62
Output dim: 6, lower bound: -0.0073069, upper bound: 0.0072700
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.62
Output dim: 6, lower bound: -0.0072545, upper bound: 0.0072894
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.62
Output dim: 6, lower bound: -0.0072520, upper bound: 0.0072945
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.62
Output dim: 6, lower bound: -0.0060562, upper bound: 0.0060550
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.62
Output dim: 6, lower bound: -0.0060562, upper bound: 0.0060550
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.62
Output dim: 6, lower bound: -0.0072129, upper bound: 0.0072144
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.62
Output dim: 6, lower bound: -0.0072129, upper bound: 0.0072144

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037834, 0.0086199, 0.0037834, 0.0086199, -0.0048365, 0.0048365
1: -0.0002308, 0.0047780, -0.0002308, 0.0047780, -0.0050088, 0.0050088
2: -0.0257978, -0.0053192, -0.0257978, -0.0053192, -0.0204786, 0.0204786
3: -0.0020375, 0.0080109, -0.0020375, 0.0080109, -0.0100484, 0.0100484
4: 0.0114507, 0.0185106, 0.0114507, 0.0185106, -0.0070599, 0.0070599
5: -0.0032830, 0.0100317, -0.0032830, 0.0100317, -0.0133147, 0.0133147
6: 0.9943414, 1.0042117, 0.9943414, 1.0042117, -0.0098702, 0.0098702
7: 0.0073448, 0.0201244, 0.0073448, 0.0201244, -0.0094426, 0.0093669
8: 0.0018158, 0.0072932, 0.0018158, 0.0072932, -0.0054774, 0.0054774
9: -0.0276300, -0.0138944, -0.0276300, -0.0138944, -0.0137356, 0.0137356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0072220, upper bound: 0.0071789
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0072168, upper bound: 0.0071791
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037834, 0.0086199, 0.0037834, 0.0086199, -0.0048365, 0.0048365
1: -0.0002308, 0.0047780, -0.0002308, 0.0047780, -0.0050088, 0.0050088
2: -0.0257978, -0.0053192, -0.0257978, -0.0053192, -0.0204786, 0.0204786
3: -0.0020375, 0.0080109, -0.0020375, 0.0080109, -0.0100484, 0.0100484
4: 0.0114507, 0.0185106, 0.0114507, 0.0185106, -0.0070599, 0.0070599
5: -0.0032830, 0.0100317, -0.0032830, 0.0100317, -0.0133147, 0.0133147
6: 0.9943414, 1.0042117, 0.9943414, 1.0042117, -0.0098702, 0.0098702
7: 0.0073448, 0.0201244, 0.0073448, 0.0201244, -0.0094135, 0.0093910
8: 0.0018158, 0.0072932, 0.0018158, 0.0072932, -0.0054774, 0.0054774
9: -0.0276300, -0.0138944, -0.0276300, -0.0138944, -0.0137356, 0.0137356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 153

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073036, upper bound: 0.0072658
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0073033, upper bound: 0.0072658
time: 1.13 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037834, 0.0086199, 0.0037834, 0.0086199, -0.0048365, 0.0048365
1: -0.0002308, 0.0047780, -0.0002308, 0.0047780, -0.0050088, 0.0050088
2: -0.0257978, -0.0053192, -0.0257978, -0.0053192, -0.0204786, 0.0204786
3: -0.0020375, 0.0080109, -0.0020375, 0.0080109, -0.0100484, 0.0100484
4: 0.0114507, 0.0185106, 0.0114507, 0.0185106, -0.0070599, 0.0070599
5: -0.0032830, 0.0100317, -0.0032830, 0.0100317, -0.0133147, 0.0133147
6: 0.9943414, 1.0042117, 0.9943414, 1.0042117, -0.0098702, 0.0098702
7: 0.0073448, 0.0201244, 0.0073448, 0.0201244, -0.0092680, 0.0093202
8: 0.0018158, 0.0072932, 0.0018158, 0.0072932, -0.0054774, 0.0054774
9: -0.0276300, -0.0138944, -0.0276300, -0.0138944, -0.0137356, 0.0137356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 153

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0072038, upper bound: 0.0072386
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0072038, upper bound: 0.0072386
time: 1.17 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037834, 0.0086199, 0.0037834, 0.0086199, -0.0048365, 0.0048365
1: -0.0002308, 0.0047780, -0.0002308, 0.0047780, -0.0050088, 0.0050088
2: -0.0257978, -0.0053192, -0.0257978, -0.0053192, -0.0204786, 0.0204786
3: -0.0020375, 0.0080109, -0.0020375, 0.0080109, -0.0100484, 0.0100484
4: 0.0114507, 0.0185106, 0.0114507, 0.0185106, -0.0070599, 0.0070599
5: -0.0032830, 0.0100317, -0.0032830, 0.0100317, -0.0133147, 0.0133147
6: 0.9943414, 1.0042117, 0.9943414, 1.0042117, -0.0098702, 0.0098702
7: 0.0073448, 0.0201244, 0.0073448, 0.0201244, -0.0092663, 0.0093220
8: 0.0018158, 0.0072932, 0.0018158, 0.0072932, -0.0054774, 0.0054774
9: -0.0276300, -0.0138944, -0.0276300, -0.0138944, -0.0137356, 0.0137356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 165

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0071620, upper bound: 0.0071967
time: 1.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0071582, upper bound: 0.0072066
time: 1.27 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037834, 0.0086199, 0.0037834, 0.0086199, -0.0048365, 0.0048365
1: -0.0002308, 0.0047780, -0.0002308, 0.0047780, -0.0050088, 0.0050088
2: -0.0257978, -0.0053192, -0.0257978, -0.0053192, -0.0204786, 0.0204786
3: -0.0020375, 0.0080109, -0.0020375, 0.0080109, -0.0100484, 0.0100484
4: 0.0114507, 0.0185106, 0.0114507, 0.0185106, -0.0070599, 0.0070599
5: -0.0032830, 0.0100317, -0.0032830, 0.0100317, -0.0133147, 0.0133147
6: 0.9943414, 1.0042117, 0.9943414, 1.0042117, -0.0098702, 0.0098702
7: 0.0073448, 0.0201244, 0.0073448, 0.0201244, -0.0093821, 0.0093841
8: 0.0018158, 0.0072932, 0.0018158, 0.0072932, -0.0054774, 0.0054774
9: -0.0276300, -0.0138944, -0.0276300, -0.0138944, -0.0137356, 0.0137356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 188

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0071963, upper bound: 0.0071947
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0071928, upper bound: 0.0071978
time: 1.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037834, 0.0086199, 0.0037834, 0.0086199, -0.0048365, 0.0048365
1: -0.0002308, 0.0047780, -0.0002308, 0.0047780, -0.0050088, 0.0050088
2: -0.0257978, -0.0053192, -0.0257978, -0.0053192, -0.0204786, 0.0204786
3: -0.0020375, 0.0080109, -0.0020375, 0.0080109, -0.0100484, 0.0100484
4: 0.0114507, 0.0185106, 0.0114507, 0.0185106, -0.0070599, 0.0070599
5: -0.0032830, 0.0100317, -0.0032830, 0.0100317, -0.0133147, 0.0133147
6: 0.9943414, 1.0042117, 0.9943414, 1.0042117, -0.0098702, 0.0098702
7: 0.0073448, 0.0201244, 0.0073448, 0.0201244, -0.0093795, 0.0093919
8: 0.0018158, 0.0072932, 0.0018158, 0.0072932, -0.0054774, 0.0054774
9: -0.0276300, -0.0138944, -0.0276300, -0.0138944, -0.0137356, 0.0137356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0072060, upper bound: 0.0072072
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0072059, upper bound: 0.0072075
time: 1.05 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.59 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.59
Output dim: 6, lower bound: -0.0072220, upper bound: 0.0071789
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.59
Output dim: 6, lower bound: -0.0072168, upper bound: 0.0071791
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.59
Output dim: 6, lower bound: -0.0073036, upper bound: 0.0072658
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.59
Output dim: 6, lower bound: -0.0073033, upper bound: 0.0072658
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.59
Output dim: 6, lower bound: -0.0072038, upper bound: 0.0072386
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.59
Output dim: 6, lower bound: -0.0072038, upper bound: 0.0072386
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.59
Output dim: 6, lower bound: -0.0071620, upper bound: 0.0071967
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.59
Output dim: 6, lower bound: -0.0071582, upper bound: 0.0072066
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.59
Output dim: 6, lower bound: -0.0071963, upper bound: 0.0071947
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.59
Output dim: 6, lower bound: -0.0071928, upper bound: 0.0071978
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.59
Output dim: 6, lower bound: -0.0072060, upper bound: 0.0072072
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.59
Output dim: 6, lower bound: -0.0072059, upper bound: 0.0072075

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037834, 0.0086199, 0.0037834, 0.0086199, -0.0048365, 0.0048365
1: -0.0002308, 0.0047780, -0.0002308, 0.0047780, -0.0050088, 0.0050088
2: -0.0257978, -0.0053192, -0.0257978, -0.0053192, -0.0204786, 0.0204786
3: -0.0020375, 0.0080109, -0.0020375, 0.0080109, -0.0100484, 0.0100484
4: 0.0114507, 0.0185106, 0.0114507, 0.0185106, -0.0070599, 0.0070599
5: -0.0032830, 0.0100317, -0.0032830, 0.0100317, -0.0133147, 0.0133147
6: 0.9943414, 1.0042117, 0.9943414, 1.0042117, -0.0098702, 0.0098702
7: 0.0073448, 0.0201244, 0.0073448, 0.0201244, -0.0093576, 0.0092602
8: 0.0018158, 0.0072932, 0.0018158, 0.0072932, -0.0054774, 0.0054774
9: -0.0276300, -0.0138944, -0.0276300, -0.0138944, -0.0137356, 0.0137356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 75

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0067413, upper bound: 0.0067107
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0067413, upper bound: 0.0067107
time: 1.08 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037834, 0.0086199, 0.0037834, 0.0086199, -0.0048365, 0.0048365
1: -0.0002308, 0.0047780, -0.0002308, 0.0047780, -0.0050088, 0.0050088
2: -0.0257978, -0.0053192, -0.0257978, -0.0053192, -0.0204786, 0.0204786
3: -0.0020375, 0.0080109, -0.0020375, 0.0080109, -0.0100484, 0.0100484
4: 0.0114507, 0.0185106, 0.0114507, 0.0185106, -0.0070599, 0.0070599
5: -0.0032830, 0.0100317, -0.0032830, 0.0100317, -0.0133147, 0.0133147
6: 0.9943414, 1.0042117, 0.9943414, 1.0042117, -0.0098702, 0.0098702
7: 0.0073448, 0.0201244, 0.0073448, 0.0201244, -0.0093360, 0.0092749
8: 0.0018158, 0.0072932, 0.0018158, 0.0072932, -0.0054774, 0.0054774
9: -0.0276300, -0.0138944, -0.0276300, -0.0138944, -0.0137356, 0.0137356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0071301, upper bound: 0.0070850
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0071187, upper bound: 0.0070906
time: 1.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037834, 0.0086199, 0.0037834, 0.0086199, -0.0048365, 0.0048365
1: -0.0002308, 0.0047780, -0.0002308, 0.0047780, -0.0050088, 0.0050088
2: -0.0257978, -0.0053192, -0.0257978, -0.0053192, -0.0204786, 0.0204786
3: -0.0020375, 0.0080109, -0.0020375, 0.0080109, -0.0100484, 0.0100484
4: 0.0114507, 0.0185106, 0.0114507, 0.0185106, -0.0070599, 0.0070599
5: -0.0032830, 0.0100317, -0.0032830, 0.0100317, -0.0133147, 0.0133147
6: 0.9943414, 1.0042117, 0.9943414, 1.0042117, -0.0098702, 0.0098702
7: 0.0073448, 0.0201244, 0.0073448, 0.0201244, -0.0093730, 0.0093548
8: 0.0018158, 0.0072932, 0.0018158, 0.0072932, -0.0054774, 0.0054774
9: -0.0276300, -0.0138944, -0.0276300, -0.0138944, -0.0137356, 0.0137356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 188

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0072865, upper bound: 0.0072470
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0072801, upper bound: 0.0072490
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037834, 0.0086199, 0.0037834, 0.0086199, -0.0048365, 0.0048365
1: -0.0002308, 0.0047780, -0.0002308, 0.0047780, -0.0050088, 0.0050088
2: -0.0257978, -0.0053192, -0.0257978, -0.0053192, -0.0204786, 0.0204786
3: -0.0020375, 0.0080109, -0.0020375, 0.0080109, -0.0100484, 0.0100484
4: 0.0114507, 0.0185106, 0.0114507, 0.0185106, -0.0070599, 0.0070599
5: -0.0032830, 0.0100317, -0.0032830, 0.0100317, -0.0133147, 0.0133147
6: 0.9943414, 1.0042117, 0.9943414, 1.0042117, -0.0098702, 0.0098702
7: 0.0073448, 0.0201244, 0.0073448, 0.0201244, -0.0093739, 0.0093504
8: 0.0018158, 0.0072932, 0.0018158, 0.0072932, -0.0054774, 0.0054774
9: -0.0276300, -0.0138944, -0.0276300, -0.0138944, -0.0137356, 0.0137356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 188

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0072862, upper bound: 0.0072474
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0072799, upper bound: 0.0072490
time: 1.43 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037834, 0.0086199, 0.0037834, 0.0086199, -0.0048365, 0.0048365
1: -0.0002308, 0.0047780, -0.0002308, 0.0047780, -0.0050088, 0.0050088
2: -0.0257978, -0.0053192, -0.0257978, -0.0053192, -0.0204786, 0.0204786
3: -0.0020375, 0.0080109, -0.0020375, 0.0080109, -0.0100484, 0.0100484
4: 0.0114507, 0.0185106, 0.0114507, 0.0185106, -0.0070599, 0.0070599
5: -0.0032830, 0.0100317, -0.0032830, 0.0100317, -0.0133147, 0.0133147
6: 0.9943414, 1.0042117, 0.9943414, 1.0042117, -0.0098702, 0.0098702
7: 0.0073448, 0.0201244, 0.0073448, 0.0201244, -0.0091513, 0.0092010
8: 0.0018158, 0.0072932, 0.0018158, 0.0072932, -0.0054774, 0.0054774
9: -0.0276300, -0.0138944, -0.0276300, -0.0138944, -0.0137356, 0.0137356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0065391, upper bound: 0.0065618
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0065391, upper bound: 0.0065618
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037834, 0.0086199, 0.0037834, 0.0086199, -0.0048365, 0.0048365
1: -0.0002308, 0.0047780, -0.0002308, 0.0047780, -0.0050088, 0.0050088
2: -0.0257978, -0.0053192, -0.0257978, -0.0053192, -0.0204786, 0.0204786
3: -0.0020375, 0.0080109, -0.0020375, 0.0080109, -0.0100484, 0.0100484
4: 0.0114507, 0.0185106, 0.0114507, 0.0185106, -0.0070599, 0.0070599
5: -0.0032830, 0.0100317, -0.0032830, 0.0100317, -0.0133147, 0.0133147
6: 0.9943414, 1.0042117, 0.9943414, 1.0042117, -0.0098702, 0.0098702
7: 0.0073448, 0.0201244, 0.0073448, 0.0201244, -0.0091487, 0.0092035
8: 0.0018158, 0.0072932, 0.0018158, 0.0072932, -0.0054774, 0.0054774
9: -0.0276300, -0.0138944, -0.0276300, -0.0138944, -0.0137356, 0.0137356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 165

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 75

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0065986, upper bound: 0.0066151
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0065986, upper bound: 0.0066151
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037834, 0.0086199, 0.0037834, 0.0086199, -0.0048365, 0.0048365
1: -0.0002308, 0.0047780, -0.0002308, 0.0047780, -0.0050088, 0.0050088
2: -0.0257978, -0.0053192, -0.0257978, -0.0053192, -0.0204786, 0.0204786
3: -0.0020375, 0.0080109, -0.0020375, 0.0080109, -0.0100484, 0.0100484
4: 0.0114507, 0.0185106, 0.0114507, 0.0185106, -0.0070599, 0.0070599
5: -0.0032830, 0.0100317, -0.0032830, 0.0100317, -0.0133147, 0.0133147
6: 0.9943414, 1.0042117, 0.9943414, 1.0042117, -0.0098702, 0.0098702
7: 0.0073448, 0.0201244, 0.0073448, 0.0201244, -0.0091346, 0.0091772
8: 0.0018158, 0.0072932, 0.0018158, 0.0072932, -0.0054774, 0.0054774
9: -0.0276300, -0.0138944, -0.0276300, -0.0138944, -0.0137356, 0.0137356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0066559, upper bound: 0.0066718
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0066559, upper bound: 0.0066718
time: 1.11 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037834, 0.0086199, 0.0037834, 0.0086199, -0.0048365, 0.0048365
1: -0.0002308, 0.0047780, -0.0002308, 0.0047780, -0.0050088, 0.0050088
2: -0.0257978, -0.0053192, -0.0257978, -0.0053192, -0.0204786, 0.0204786
3: -0.0020375, 0.0080109, -0.0020375, 0.0080109, -0.0100484, 0.0100484
4: 0.0114507, 0.0185106, 0.0114507, 0.0185106, -0.0070599, 0.0070599
5: -0.0032830, 0.0100317, -0.0032830, 0.0100317, -0.0133147, 0.0133147
6: 0.9943414, 1.0042117, 0.9943414, 1.0042117, -0.0098702, 0.0098702
7: 0.0073448, 0.0201244, 0.0073448, 0.0201244, -0.0091215, 0.0091947
8: 0.0018158, 0.0072932, 0.0018158, 0.0072932, -0.0054774, 0.0054774
9: -0.0276300, -0.0138944, -0.0276300, -0.0138944, -0.0137356, 0.0137356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0071061, upper bound: 0.0071503
time: 1.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0071061, upper bound: 0.0071503
time: 1.50 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037834, 0.0086199, 0.0037834, 0.0086199, -0.0048365, 0.0048365
1: -0.0002308, 0.0047780, -0.0002308, 0.0047780, -0.0050088, 0.0050088
2: -0.0257978, -0.0053192, -0.0257978, -0.0053192, -0.0204786, 0.0204786
3: -0.0020375, 0.0080109, -0.0020375, 0.0080109, -0.0100484, 0.0100484
4: 0.0114507, 0.0185106, 0.0114507, 0.0185106, -0.0070599, 0.0070599
5: -0.0032830, 0.0100317, -0.0032830, 0.0100317, -0.0133147, 0.0133147
6: 0.9943414, 1.0042117, 0.9943414, 1.0042117, -0.0098702, 0.0098702
7: 0.0073448, 0.0201244, 0.0073448, 0.0201244, -0.0093910, 0.0093919
8: 0.0018158, 0.0072932, 0.0018158, 0.0072932, -0.0054774, 0.0054774
9: -0.0276300, -0.0138944, -0.0276300, -0.0138944, -0.0137356, 0.0137356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0071048, upper bound: 0.0070930
time: 1.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0070917, upper bound: 0.0071029
time: 1.19 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037834, 0.0086199, 0.0037834, 0.0086199, -0.0048365, 0.0048365
1: -0.0002308, 0.0047780, -0.0002308, 0.0047780, -0.0050088, 0.0050088
2: -0.0257978, -0.0053192, -0.0257978, -0.0053192, -0.0204786, 0.0204786
3: -0.0020375, 0.0080109, -0.0020375, 0.0080109, -0.0100484, 0.0100484
4: 0.0114507, 0.0185106, 0.0114507, 0.0185106, -0.0070599, 0.0070599
5: -0.0032830, 0.0100317, -0.0032830, 0.0100317, -0.0133147, 0.0133147
6: 0.9943414, 1.0042117, 0.9943414, 1.0042117, -0.0098702, 0.0098702
7: 0.0073448, 0.0201244, 0.0073448, 0.0201244, -0.0093893, 0.0093930
8: 0.0018158, 0.0072932, 0.0018158, 0.0072932, -0.0054774, 0.0054774
9: -0.0276300, -0.0138944, -0.0276300, -0.0138944, -0.0137356, 0.0137356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 165

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0071858, upper bound: 0.0071907
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0071857, upper bound: 0.0071909
time: 1.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037834, 0.0086199, 0.0037834, 0.0086199, -0.0048365, 0.0048365
1: -0.0002308, 0.0047780, -0.0002308, 0.0047780, -0.0050088, 0.0050088
2: -0.0257978, -0.0053192, -0.0257978, -0.0053192, -0.0204786, 0.0204786
3: -0.0020375, 0.0080109, -0.0020375, 0.0080109, -0.0100484, 0.0100484
4: 0.0114507, 0.0185106, 0.0114507, 0.0185106, -0.0070599, 0.0070599
5: -0.0032830, 0.0100317, -0.0032830, 0.0100317, -0.0133147, 0.0133147
6: 0.9943414, 1.0042117, 0.9943414, 1.0042117, -0.0098702, 0.0098702
7: 0.0073448, 0.0201244, 0.0073448, 0.0201244, -0.0093649, 0.0093807
8: 0.0018158, 0.0072932, 0.0018158, 0.0072932, -0.0054774, 0.0054774
9: -0.0276300, -0.0138944, -0.0276300, -0.0138944, -0.0137356, 0.0137356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0072060, upper bound: 0.0072042
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0072036, upper bound: 0.0072072
time: 1.16 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037834, 0.0086199, 0.0037834, 0.0086199, -0.0048365, 0.0048365
1: -0.0002308, 0.0047780, -0.0002308, 0.0047780, -0.0050088, 0.0050088
2: -0.0257978, -0.0053192, -0.0257978, -0.0053192, -0.0204786, 0.0204786
3: -0.0020375, 0.0080109, -0.0020375, 0.0080109, -0.0100484, 0.0100484
4: 0.0114507, 0.0185106, 0.0114507, 0.0185106, -0.0070599, 0.0070599
5: -0.0032830, 0.0100317, -0.0032830, 0.0100317, -0.0133147, 0.0133147
6: 0.9943414, 1.0042117, 0.9943414, 1.0042117, -0.0098702, 0.0098702
7: 0.0073448, 0.0201244, 0.0073448, 0.0201244, -0.0093690, 0.0093774
8: 0.0018158, 0.0072932, 0.0018158, 0.0072932, -0.0054774, 0.0054774
9: -0.0276300, -0.0138944, -0.0276300, -0.0138944, -0.0137356, 0.0137356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0072060, upper bound: 0.0072050
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0072030, upper bound: 0.0072075
time: 1.05 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.56 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.56
Output dim: 6, lower bound: -0.0067413, upper bound: 0.0067107
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.56
Output dim: 6, lower bound: -0.0067413, upper bound: 0.0067107
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.56
Output dim: 6, lower bound: -0.0071301, upper bound: 0.0070850
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.56
Output dim: 6, lower bound: -0.0071187, upper bound: 0.0070906
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.56
Output dim: 6, lower bound: -0.0072865, upper bound: 0.0072470
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.56
Output dim: 6, lower bound: -0.0072801, upper bound: 0.0072490
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.56
Output dim: 6, lower bound: -0.0072862, upper bound: 0.0072474
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.56
Output dim: 6, lower bound: -0.0072799, upper bound: 0.0072490
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.56
Output dim: 6, lower bound: -0.0065391, upper bound: 0.0065618
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.56
Output dim: 6, lower bound: -0.0065391, upper bound: 0.0065618
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.56
Output dim: 6, lower bound: -0.0065986, upper bound: 0.0066151
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.56
Output dim: 6, lower bound: -0.0065986, upper bound: 0.0066151
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.56
Output dim: 6, lower bound: -0.0066559, upper bound: 0.0066718
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.56
Output dim: 6, lower bound: -0.0066559, upper bound: 0.0066718
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.56
Output dim: 6, lower bound: -0.0071061, upper bound: 0.0071503
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.56
Output dim: 6, lower bound: -0.0071061, upper bound: 0.0071503
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.56
Output dim: 6, lower bound: -0.0071048, upper bound: 0.0070930
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.56
Output dim: 6, lower bound: -0.0070917, upper bound: 0.0071029
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.56
Output dim: 6, lower bound: -0.0071858, upper bound: 0.0071907
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.56
Output dim: 6, lower bound: -0.0071857, upper bound: 0.0071909
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.56
Output dim: 6, lower bound: -0.0072060, upper bound: 0.0072042
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.56
Output dim: 6, lower bound: -0.0072036, upper bound: 0.0072072
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.56
Output dim: 6, lower bound: -0.0072060, upper bound: 0.0072050
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.56
Output dim: 6, lower bound: -0.0072030, upper bound: 0.0072075

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037834, 0.0086199, 0.0037834, 0.0086199, -0.0048365, 0.0048365
1: -0.0002308, 0.0047780, -0.0002308, 0.0047780, -0.0050088, 0.0050088
2: -0.0257978, -0.0053192, -0.0257978, -0.0053192, -0.0204786, 0.0204786
3: -0.0020375, 0.0080109, -0.0020375, 0.0080109, -0.0100484, 0.0100484
4: 0.0114507, 0.0185106, 0.0114507, 0.0185106, -0.0070599, 0.0070599
5: -0.0032830, 0.0100317, -0.0032830, 0.0100317, -0.0133147, 0.0133147
6: 0.9943414, 1.0042117, 0.9943414, 1.0042117, -0.0098702, 0.0098702
7: 0.0073448, 0.0201244, 0.0073448, 0.0201244, -0.0092124, 0.0091339
8: 0.0018158, 0.0072932, 0.0018158, 0.0072932, -0.0054774, 0.0054774
9: -0.0276300, -0.0138944, -0.0276300, -0.0138944, -0.0137356, 0.0137356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0066074, upper bound: 0.0065764
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0066074, upper bound: 0.0065764
time: 1.17 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037834, 0.0086199, 0.0037834, 0.0086199, -0.0048365, 0.0048365
1: -0.0002308, 0.0047780, -0.0002308, 0.0047780, -0.0050088, 0.0050088
2: -0.0257978, -0.0053192, -0.0257978, -0.0053192, -0.0204786, 0.0204786
3: -0.0020375, 0.0080109, -0.0020375, 0.0080109, -0.0100484, 0.0100484
4: 0.0114507, 0.0185106, 0.0114507, 0.0185106, -0.0070599, 0.0070599
5: -0.0032830, 0.0100317, -0.0032830, 0.0100317, -0.0133147, 0.0133147
6: 0.9943414, 1.0042117, 0.9943414, 1.0042117, -0.0098702, 0.0098702
7: 0.0073448, 0.0201244, 0.0073448, 0.0201244, -0.0091949, 0.0091475
8: 0.0018158, 0.0072932, 0.0018158, 0.0072932, -0.0054774, 0.0054774
9: -0.0276300, -0.0138944, -0.0276300, -0.0138944, -0.0137356, 0.0137356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 165

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0065949, upper bound: 0.0065812
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0065949, upper bound: 0.0065812
time: 1.09 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037834, 0.0086199, 0.0037834, 0.0086199, -0.0048365, 0.0048365
1: -0.0002308, 0.0047780, -0.0002308, 0.0047780, -0.0050088, 0.0050088
2: -0.0257978, -0.0053192, -0.0257978, -0.0053192, -0.0204786, 0.0204786
3: -0.0020375, 0.0080109, -0.0020375, 0.0080109, -0.0100484, 0.0100484
4: 0.0114507, 0.0185106, 0.0114507, 0.0185106, -0.0070599, 0.0070599
5: -0.0032830, 0.0100317, -0.0032830, 0.0100317, -0.0133147, 0.0133147
6: 0.9943414, 1.0042117, 0.9943414, 1.0042117, -0.0098702, 0.0098702
7: 0.0073448, 0.0201244, 0.0073448, 0.0201244, -0.0093796, 0.0093600
8: 0.0018158, 0.0072932, 0.0018158, 0.0072932, -0.0054774, 0.0054774
9: -0.0276300, -0.0138944, -0.0276300, -0.0138944, -0.0137356, 0.0137356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0071985, upper bound: 0.0071536
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0071879, upper bound: 0.0071573
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037834, 0.0086199, 0.0037834, 0.0086199, -0.0048365, 0.0048365
1: -0.0002308, 0.0047780, -0.0002308, 0.0047780, -0.0050088, 0.0050088
2: -0.0257978, -0.0053192, -0.0257978, -0.0053192, -0.0204786, 0.0204786
3: -0.0020375, 0.0080109, -0.0020375, 0.0080109, -0.0100484, 0.0100484
4: 0.0114507, 0.0185106, 0.0114507, 0.0185106, -0.0070599, 0.0070599
5: -0.0032830, 0.0100317, -0.0032830, 0.0100317, -0.0133147, 0.0133147
6: 0.9943414, 1.0042117, 0.9943414, 1.0042117, -0.0098702, 0.0098702
7: 0.0073448, 0.0201244, 0.0073448, 0.0201244, -0.0093777, 0.0093614
8: 0.0018158, 0.0072932, 0.0018158, 0.0072932, -0.0054774, 0.0054774
9: -0.0276300, -0.0138944, -0.0276300, -0.0138944, -0.0137356, 0.0137356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0071918, upper bound: 0.0071540
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0071853, upper bound: 0.0071596
time: 1.17 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037834, 0.0086199, 0.0037834, 0.0086199, -0.0048365, 0.0048365
1: -0.0002308, 0.0047780, -0.0002308, 0.0047780, -0.0050088, 0.0050088
2: -0.0257978, -0.0053192, -0.0257978, -0.0053192, -0.0204786, 0.0204786
3: -0.0020375, 0.0080109, -0.0020375, 0.0080109, -0.0100484, 0.0100484
4: 0.0114507, 0.0185106, 0.0114507, 0.0185106, -0.0070599, 0.0070599
5: -0.0032830, 0.0100317, -0.0032830, 0.0100317, -0.0133147, 0.0133147
6: 0.9943414, 1.0042117, 0.9943414, 1.0042117, -0.0098702, 0.0098702
7: 0.0073448, 0.0201244, 0.0073448, 0.0201244, -0.0093805, 0.0093556
8: 0.0018158, 0.0072932, 0.0018158, 0.0072932, -0.0054774, 0.0054774
9: -0.0276300, -0.0138944, -0.0276300, -0.0138944, -0.0137356, 0.0137356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 75

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0068112, upper bound: 0.0067916
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0068112, upper bound: 0.0067916
time: 1.15 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037834, 0.0086199, 0.0037834, 0.0086199, -0.0048365, 0.0048365
1: -0.0002308, 0.0047780, -0.0002308, 0.0047780, -0.0050088, 0.0050088
2: -0.0257978, -0.0053192, -0.0257978, -0.0053192, -0.0204786, 0.0204786
3: -0.0020375, 0.0080109, -0.0020375, 0.0080109, -0.0100484, 0.0100484
4: 0.0114507, 0.0185106, 0.0114507, 0.0185106, -0.0070599, 0.0070599
5: -0.0032830, 0.0100317, -0.0032830, 0.0100317, -0.0133147, 0.0133147
6: 0.9943414, 1.0042117, 0.9943414, 1.0042117, -0.0098702, 0.0098702
7: 0.0073448, 0.0201244, 0.0073448, 0.0201244, -0.0093786, 0.0093571
8: 0.0018158, 0.0072932, 0.0018158, 0.0072932, -0.0054774, 0.0054774
9: -0.0276300, -0.0138944, -0.0276300, -0.0138944, -0.0137356, 0.0137356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0072230, upper bound: 0.0071953
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0072230, upper bound: 0.0071953
time: 1.14 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037834, 0.0086199, 0.0037834, 0.0086199, -0.0048365, 0.0048365
1: -0.0002308, 0.0047780, -0.0002308, 0.0047780, -0.0050088, 0.0050088
2: -0.0257978, -0.0053192, -0.0257978, -0.0053192, -0.0204786, 0.0204786
3: -0.0020375, 0.0080109, -0.0020375, 0.0080109, -0.0100484, 0.0100484
4: 0.0114507, 0.0185106, 0.0114507, 0.0185106, -0.0070599, 0.0070599
5: -0.0032830, 0.0100317, -0.0032830, 0.0100317, -0.0133147, 0.0133147
6: 0.9943414, 1.0042117, 0.9943414, 1.0042117, -0.0098702, 0.0098702
7: 0.0073448, 0.0201244, 0.0073448, 0.0201244, -0.0090038, 0.0090754
8: 0.0018158, 0.0072932, 0.0018158, 0.0072932, -0.0054774, 0.0054774
9: -0.0276300, -0.0138944, -0.0276300, -0.0138944, -0.0137356, 0.0137356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0070209, upper bound: 0.0070646
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0070207, upper bound: 0.0070660
time: 1.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037834, 0.0086199, 0.0037834, 0.0086199, -0.0048365, 0.0048365
1: -0.0002308, 0.0047780, -0.0002308, 0.0047780, -0.0050088, 0.0050088
2: -0.0257978, -0.0053192, -0.0257978, -0.0053192, -0.0204786, 0.0204786
3: -0.0020375, 0.0080109, -0.0020375, 0.0080109, -0.0100484, 0.0100484
4: 0.0114507, 0.0185106, 0.0114507, 0.0185106, -0.0070599, 0.0070599
5: -0.0032830, 0.0100317, -0.0032830, 0.0100317, -0.0133147, 0.0133147
6: 0.9943414, 1.0042117, 0.9943414, 1.0042117, -0.0098702, 0.0098702
7: 0.0073448, 0.0201244, 0.0073448, 0.0201244, -0.0090013, 0.0090769
8: 0.0018158, 0.0072932, 0.0018158, 0.0072932, -0.0054774, 0.0054774
9: -0.0276300, -0.0138944, -0.0276300, -0.0138944, -0.0137356, 0.0137356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0064560, upper bound: 0.0064868
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0064560, upper bound: 0.0064868
time: 1.15 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037834, 0.0086199, 0.0037834, 0.0086199, -0.0048365, 0.0048365
1: -0.0002308, 0.0047780, -0.0002308, 0.0047780, -0.0050088, 0.0050088
2: -0.0257978, -0.0053192, -0.0257978, -0.0053192, -0.0204786, 0.0204786
3: -0.0020375, 0.0080109, -0.0020375, 0.0080109, -0.0100484, 0.0100484
4: 0.0114507, 0.0185106, 0.0114507, 0.0185106, -0.0070599, 0.0070599
5: -0.0032830, 0.0100317, -0.0032830, 0.0100317, -0.0133147, 0.0133147
6: 0.9943414, 1.0042117, 0.9943414, 1.0042117, -0.0098702, 0.0098702
7: 0.0073448, 0.0201244, 0.0073448, 0.0201244, -0.0092565, 0.0092443
8: 0.0018158, 0.0072932, 0.0018158, 0.0072932, -0.0054774, 0.0054774
9: -0.0276300, -0.0138944, -0.0276300, -0.0138944, -0.0137356, 0.0137356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0064629, upper bound: 0.0064498
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0064629, upper bound: 0.0064498
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037834, 0.0086199, 0.0037834, 0.0086199, -0.0048365, 0.0048365
1: -0.0002308, 0.0047780, -0.0002308, 0.0047780, -0.0050088, 0.0050088
2: -0.0257978, -0.0053192, -0.0257978, -0.0053192, -0.0204786, 0.0204786
3: -0.0020375, 0.0080109, -0.0020375, 0.0080109, -0.0100484, 0.0100484
4: 0.0114507, 0.0185106, 0.0114507, 0.0185106, -0.0070599, 0.0070599
5: -0.0032830, 0.0100317, -0.0032830, 0.0100317, -0.0133147, 0.0133147
6: 0.9943414, 1.0042117, 0.9943414, 1.0042117, -0.0098702, 0.0098702
7: 0.0073448, 0.0201244, 0.0073448, 0.0201244, -0.0092432, 0.0092588
8: 0.0018158, 0.0072932, 0.0018158, 0.0072932, -0.0054774, 0.0054774
9: -0.0276300, -0.0138944, -0.0276300, -0.0138944, -0.0137356, 0.0137356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0064503, upper bound: 0.0064609
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0064503, upper bound: 0.0064609
time: 1.08 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037834, 0.0086199, 0.0037834, 0.0086199, -0.0048365, 0.0048365
1: -0.0002308, 0.0047780, -0.0002308, 0.0047780, -0.0050088, 0.0050088
2: -0.0257978, -0.0053192, -0.0257978, -0.0053192, -0.0204786, 0.0204786
3: -0.0020375, 0.0080109, -0.0020375, 0.0080109, -0.0100484, 0.0100484
4: 0.0114507, 0.0185106, 0.0114507, 0.0185106, -0.0070599, 0.0070599
5: -0.0032830, 0.0100317, -0.0032830, 0.0100317, -0.0133147, 0.0133147
6: 0.9943414, 1.0042117, 0.9943414, 1.0042117, -0.0098702, 0.0098702
7: 0.0073448, 0.0201244, 0.0073448, 0.0201244, -0.0093742, 0.0093823
8: 0.0018158, 0.0072932, 0.0018158, 0.0072932, -0.0054774, 0.0054774
9: -0.0276300, -0.0138944, -0.0276300, -0.0138944, -0.0137356, 0.0137356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0069155, upper bound: 0.0069206
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0069155, upper bound: 0.0069206
time: 1.11 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037834, 0.0086199, 0.0037834, 0.0086199, -0.0048365, 0.0048365
1: -0.0002308, 0.0047780, -0.0002308, 0.0047780, -0.0050088, 0.0050088
2: -0.0257978, -0.0053192, -0.0257978, -0.0053192, -0.0204786, 0.0204786
3: -0.0020375, 0.0080109, -0.0020375, 0.0080109, -0.0100484, 0.0100484
4: 0.0114507, 0.0185106, 0.0114507, 0.0185106, -0.0070599, 0.0070599
5: -0.0032830, 0.0100317, -0.0032830, 0.0100317, -0.0133147, 0.0133147
6: 0.9943414, 1.0042117, 0.9943414, 1.0042117, -0.0098702, 0.0098702
7: 0.0073448, 0.0201244, 0.0073448, 0.0201244, -0.0093785, 0.0093778
8: 0.0018158, 0.0072932, 0.0018158, 0.0072932, -0.0054774, 0.0054774
9: -0.0276300, -0.0138944, -0.0276300, -0.0138944, -0.0137356, 0.0137356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 75

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0065744, upper bound: 0.0065780
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0065744, upper bound: 0.0065780
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037834, 0.0086199, 0.0037834, 0.0086199, -0.0048365, 0.0048365
1: -0.0002308, 0.0047780, -0.0002308, 0.0047780, -0.0050088, 0.0050088
2: -0.0257978, -0.0053192, -0.0257978, -0.0053192, -0.0204786, 0.0204786
3: -0.0020375, 0.0080109, -0.0020375, 0.0080109, -0.0100484, 0.0100484
4: 0.0114507, 0.0185106, 0.0114507, 0.0185106, -0.0070599, 0.0070599
5: -0.0032830, 0.0100317, -0.0032830, 0.0100317, -0.0133147, 0.0133147
6: 0.9943414, 1.0042117, 0.9943414, 1.0042117, -0.0098702, 0.0098702
7: 0.0073448, 0.0201244, 0.0073448, 0.0201244, -0.0094492, 0.0094378
8: 0.0018158, 0.0072932, 0.0018158, 0.0072932, -0.0054774, 0.0054774
9: -0.0276300, -0.0138944, -0.0276300, -0.0138944, -0.0137356, 0.0137356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0071144, upper bound: 0.0070997
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0071013, upper bound: 0.0071124
time: 1.17 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037834, 0.0086199, 0.0037834, 0.0086199, -0.0048365, 0.0048365
1: -0.0002308, 0.0047780, -0.0002308, 0.0047780, -0.0050088, 0.0050088
2: -0.0257978, -0.0053192, -0.0257978, -0.0053192, -0.0204786, 0.0204786
3: -0.0020375, 0.0080109, -0.0020375, 0.0080109, -0.0100484, 0.0100484
4: 0.0114507, 0.0185106, 0.0114507, 0.0185106, -0.0070599, 0.0070599
5: -0.0032830, 0.0100317, -0.0032830, 0.0100317, -0.0133147, 0.0133147
6: 0.9943414, 1.0042117, 0.9943414, 1.0042117, -0.0098702, 0.0098702
7: 0.0073448, 0.0201244, 0.0073448, 0.0201244, -0.0094211, 0.0094652
8: 0.0018158, 0.0072932, 0.0018158, 0.0072932, -0.0054774, 0.0054774
9: -0.0276300, -0.0138944, -0.0276300, -0.0138944, -0.0137356, 0.0137356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0071117, upper bound: 0.0071024
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0071001, upper bound: 0.0071157
time: 1.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037834, 0.0086199, 0.0037834, 0.0086199, -0.0048365, 0.0048365
1: -0.0002308, 0.0047780, -0.0002308, 0.0047780, -0.0050088, 0.0050088
2: -0.0257978, -0.0053192, -0.0257978, -0.0053192, -0.0204786, 0.0204786
3: -0.0020375, 0.0080109, -0.0020375, 0.0080109, -0.0100484, 0.0100484
4: 0.0114507, 0.0185106, 0.0114507, 0.0185106, -0.0070599, 0.0070599
5: -0.0032830, 0.0100317, -0.0032830, 0.0100317, -0.0133147, 0.0133147
6: 0.9943414, 1.0042117, 0.9943414, 1.0042117, -0.0098702, 0.0098702
7: 0.0073448, 0.0201244, 0.0073448, 0.0201244, -0.0094533, 0.0094371
8: 0.0018158, 0.0072932, 0.0018158, 0.0072932, -0.0054774, 0.0054774
9: -0.0276300, -0.0138944, -0.0276300, -0.0138944, -0.0137356, 0.0137356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 165

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0071967, upper bound: 0.0071955
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0071967, upper bound: 0.0071956
time: 1.24 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037834, 0.0086199, 0.0037834, 0.0086199, -0.0048365, 0.0048365
1: -0.0002308, 0.0047780, -0.0002308, 0.0047780, -0.0050088, 0.0050088
2: -0.0257978, -0.0053192, -0.0257978, -0.0053192, -0.0204786, 0.0204786
3: -0.0020375, 0.0080109, -0.0020375, 0.0080109, -0.0100484, 0.0100484
4: 0.0114507, 0.0185106, 0.0114507, 0.0185106, -0.0070599, 0.0070599
5: -0.0032830, 0.0100317, -0.0032830, 0.0100317, -0.0133147, 0.0133147
6: 0.9943414, 1.0042117, 0.9943414, 1.0042117, -0.0098702, 0.0098702
7: 0.0073448, 0.0201244, 0.0073448, 0.0201244, -0.0094230, 0.0094618
8: 0.0018158, 0.0072932, 0.0018158, 0.0072932, -0.0054774, 0.0054774
9: -0.0276300, -0.0138944, -0.0276300, -0.0138944, -0.0137356, 0.0137356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0069298, upper bound: 0.0069357
time: 1.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0069298, upper bound: 0.0069357
time: 1.32 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 8.25 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 8.25
Output dim: 6, lower bound: -0.0066074, upper bound: 0.0065764
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 8.25
Output dim: 6, lower bound: -0.0066074, upper bound: 0.0065764
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 8.25
Output dim: 6, lower bound: -0.0065949, upper bound: 0.0065812
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 8.25
Output dim: 6, lower bound: -0.0065949, upper bound: 0.0065812
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 8.25
Output dim: 6, lower bound: -0.0071985, upper bound: 0.0071536
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 8.25
Output dim: 6, lower bound: -0.0071879, upper bound: 0.0071573
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 8.25
Output dim: 6, lower bound: -0.0071918, upper bound: 0.0071540
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 8.25
Output dim: 6, lower bound: -0.0071853, upper bound: 0.0071596
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 8.25
Output dim: 6, lower bound: -0.0068112, upper bound: 0.0067916
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 8.25
Output dim: 6, lower bound: -0.0068112, upper bound: 0.0067916
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 8.25
Output dim: 6, lower bound: -0.0072230, upper bound: 0.0071953
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 8.25
Output dim: 6, lower bound: -0.0072230, upper bound: 0.0071953
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 8.25
Output dim: 6, lower bound: -0.0070209, upper bound: 0.0070646
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 8.25
Output dim: 6, lower bound: -0.0070207, upper bound: 0.0070660
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 8.25
Output dim: 6, lower bound: -0.0064560, upper bound: 0.0064868
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 8.25
Output dim: 6, lower bound: -0.0064560, upper bound: 0.0064868
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 8.25
Output dim: 6, lower bound: -0.0064629, upper bound: 0.0064498
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 8.25
Output dim: 6, lower bound: -0.0064629, upper bound: 0.0064498
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 8.25
Output dim: 6, lower bound: -0.0064503, upper bound: 0.0064609
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 8.25
Output dim: 6, lower bound: -0.0064503, upper bound: 0.0064609
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 8.25
Output dim: 6, lower bound: -0.0069155, upper bound: 0.0069206
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 8.25
Output dim: 6, lower bound: -0.0069155, upper bound: 0.0069206
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 8.25
Output dim: 6, lower bound: -0.0065744, upper bound: 0.0065780
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 8.25
Output dim: 6, lower bound: -0.0065744, upper bound: 0.0065780
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 8.25
Output dim: 6, lower bound: -0.0071144, upper bound: 0.0070997
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 8.25
Output dim: 6, lower bound: -0.0071013, upper bound: 0.0071124
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 8.25
Output dim: 6, lower bound: -0.0071117, upper bound: 0.0071024
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 8.25
Output dim: 6, lower bound: -0.0071001, upper bound: 0.0071157
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 8.25
Output dim: 6, lower bound: -0.0071967, upper bound: 0.0071955
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 8.25
Output dim: 6, lower bound: -0.0071967, upper bound: 0.0071956
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 8.25
Output dim: 6, lower bound: -0.0069298, upper bound: 0.0069357
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 8.25
Output dim: 6, lower bound: -0.0069298, upper bound: 0.0069357

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037834, 0.0086199, 0.0037834, 0.0086199, -0.0048365, 0.0048365
1: -0.0002308, 0.0047780, -0.0002308, 0.0047780, -0.0050088, 0.0050088
2: -0.0257978, -0.0053192, -0.0257978, -0.0053192, -0.0204786, 0.0204786
3: -0.0020375, 0.0080109, -0.0020375, 0.0080109, -0.0100484, 0.0100484
4: 0.0114507, 0.0185106, 0.0114507, 0.0185106, -0.0070599, 0.0070599
5: -0.0032830, 0.0100317, -0.0032830, 0.0100317, -0.0133147, 0.0133147
6: 0.9943414, 1.0042117, 0.9943414, 1.0042117, -0.0098702, 0.0098702
7: 0.0073448, 0.0201244, 0.0073448, 0.0201244, -0.0092494, 0.0092134
8: 0.0018158, 0.0072932, 0.0018158, 0.0072932, -0.0054774, 0.0054774
9: -0.0276300, -0.0138944, -0.0276300, -0.0138944, -0.0137356, 0.0137356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0070196, upper bound: 0.0069875
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0070196, upper bound: 0.0069875
time: 1.13 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037834, 0.0086199, 0.0037834, 0.0086199, -0.0048365, 0.0048365
1: -0.0002308, 0.0047780, -0.0002308, 0.0047780, -0.0050088, 0.0050088
2: -0.0257978, -0.0053192, -0.0257978, -0.0053192, -0.0204786, 0.0204786
3: -0.0020375, 0.0080109, -0.0020375, 0.0080109, -0.0100484, 0.0100484
4: 0.0114507, 0.0185106, 0.0114507, 0.0185106, -0.0070599, 0.0070599
5: -0.0032830, 0.0100317, -0.0032830, 0.0100317, -0.0133147, 0.0133147
6: 0.9943414, 1.0042117, 0.9943414, 1.0042117, -0.0098702, 0.0098702
7: 0.0073448, 0.0201244, 0.0073448, 0.0201244, -0.0092330, 0.0092275
8: 0.0018158, 0.0072932, 0.0018158, 0.0072932, -0.0054774, 0.0054774
9: -0.0276300, -0.0138944, -0.0276300, -0.0138944, -0.0137356, 0.0137356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0071248, upper bound: 0.0071017
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0071248, upper bound: 0.0071017
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037834, 0.0086199, 0.0037834, 0.0086199, -0.0048365, 0.0048365
1: -0.0002308, 0.0047780, -0.0002308, 0.0047780, -0.0050088, 0.0050088
2: -0.0257978, -0.0053192, -0.0257978, -0.0053192, -0.0204786, 0.0204786
3: -0.0020375, 0.0080109, -0.0020375, 0.0080109, -0.0100484, 0.0100484
4: 0.0114507, 0.0185106, 0.0114507, 0.0185106, -0.0070599, 0.0070599
5: -0.0032830, 0.0100317, -0.0032830, 0.0100317, -0.0133147, 0.0133147
6: 0.9943414, 1.0042117, 0.9943414, 1.0042117, -0.0098702, 0.0098702
7: 0.0073448, 0.0201244, 0.0073448, 0.0201244, -0.0092473, 0.0092148
8: 0.0018158, 0.0072932, 0.0018158, 0.0072932, -0.0054774, 0.0054774
9: -0.0276300, -0.0138944, -0.0276300, -0.0138944, -0.0137356, 0.0137356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0071322, upper bound: 0.0070970
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0071322, upper bound: 0.0070968
time: 1.17 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037834, 0.0086199, 0.0037834, 0.0086199, -0.0048365, 0.0048365
1: -0.0002308, 0.0047780, -0.0002308, 0.0047780, -0.0050088, 0.0050088
2: -0.0257978, -0.0053192, -0.0257978, -0.0053192, -0.0204786, 0.0204786
3: -0.0020375, 0.0080109, -0.0020375, 0.0080109, -0.0100484, 0.0100484
4: 0.0114507, 0.0185106, 0.0114507, 0.0185106, -0.0070599, 0.0070599
5: -0.0032830, 0.0100317, -0.0032830, 0.0100317, -0.0133147, 0.0133147
6: 0.9943414, 1.0042117, 0.9943414, 1.0042117, -0.0098702, 0.0098702
7: 0.0073448, 0.0201244, 0.0073448, 0.0201244, -0.0092311, 0.0092288
8: 0.0018158, 0.0072932, 0.0018158, 0.0072932, -0.0054774, 0.0054774
9: -0.0276300, -0.0138944, -0.0276300, -0.0138944, -0.0137356, 0.0137356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 165

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0071756, upper bound: 0.0071500
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0071748, upper bound: 0.0071503
time: 1.23 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037834, 0.0086199, 0.0037834, 0.0086199, -0.0048365, 0.0048365
1: -0.0002308, 0.0047780, -0.0002308, 0.0047780, -0.0050088, 0.0050088
2: -0.0257978, -0.0053192, -0.0257978, -0.0053192, -0.0204786, 0.0204786
3: -0.0020375, 0.0080109, -0.0020375, 0.0080109, -0.0100484, 0.0100484
4: 0.0114507, 0.0185106, 0.0114507, 0.0185106, -0.0070599, 0.0070599
5: -0.0032830, 0.0100317, -0.0032830, 0.0100317, -0.0133147, 0.0133147
6: 0.9943414, 1.0042117, 0.9943414, 1.0042117, -0.0098702, 0.0098702
7: 0.0073448, 0.0201244, 0.0073448, 0.0201244, -0.0092523, 0.0092289
8: 0.0018158, 0.0072932, 0.0018158, 0.0072932, -0.0054774, 0.0054774
9: -0.0276300, -0.0138944, -0.0276300, -0.0138944, -0.0137356, 0.0137356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0065247, upper bound: 0.0065069
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0065247, upper bound: 0.0065069
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037834, 0.0086199, 0.0037834, 0.0086199, -0.0048365, 0.0048365
1: -0.0002308, 0.0047780, -0.0002308, 0.0047780, -0.0050088, 0.0050088
2: -0.0257978, -0.0053192, -0.0257978, -0.0053192, -0.0204786, 0.0204786
3: -0.0020375, 0.0080109, -0.0020375, 0.0080109, -0.0100484, 0.0100484
4: 0.0114507, 0.0185106, 0.0114507, 0.0185106, -0.0070599, 0.0070599
5: -0.0032830, 0.0100317, -0.0032830, 0.0100317, -0.0133147, 0.0133147
6: 0.9943414, 1.0042117, 0.9943414, 1.0042117, -0.0098702, 0.0098702
7: 0.0073448, 0.0201244, 0.0073448, 0.0201244, -0.0092470, 0.0092308
8: 0.0018158, 0.0072932, 0.0018158, 0.0072932, -0.0054774, 0.0054774
9: -0.0276300, -0.0138944, -0.0276300, -0.0138944, -0.0137356, 0.0137356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0071429, upper bound: 0.0071120
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0071425, upper bound: 0.0071122
time: 1.13 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037834, 0.0086199, 0.0037834, 0.0086199, -0.0048365, 0.0048365
1: -0.0002308, 0.0047780, -0.0002308, 0.0047780, -0.0050088, 0.0050088
2: -0.0257978, -0.0053192, -0.0257978, -0.0053192, -0.0204786, 0.0204786
3: -0.0020375, 0.0080109, -0.0020375, 0.0080109, -0.0100484, 0.0100484
4: 0.0114507, 0.0185106, 0.0114507, 0.0185106, -0.0070599, 0.0070599
5: -0.0032830, 0.0100317, -0.0032830, 0.0100317, -0.0133147, 0.0133147
6: 0.9943414, 1.0042117, 0.9943414, 1.0042117, -0.0098702, 0.0098702
7: 0.0073448, 0.0201244, 0.0073448, 0.0201244, -0.0093194, 0.0092935
8: 0.0018158, 0.0072932, 0.0018158, 0.0072932, -0.0054774, 0.0054774
9: -0.0276300, -0.0138944, -0.0276300, -0.0138944, -0.0137356, 0.0137356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 188

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0070978, upper bound: 0.0070820
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0070940, upper bound: 0.0070830
time: 1.05 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037834, 0.0086199, 0.0037834, 0.0086199, -0.0048365, 0.0048365
1: -0.0002308, 0.0047780, -0.0002308, 0.0047780, -0.0050088, 0.0050088
2: -0.0257978, -0.0053192, -0.0257978, -0.0053192, -0.0204786, 0.0204786
3: -0.0020375, 0.0080109, -0.0020375, 0.0080109, -0.0100484, 0.0100484
4: 0.0114507, 0.0185106, 0.0114507, 0.0185106, -0.0070599, 0.0070599
5: -0.0032830, 0.0100317, -0.0032830, 0.0100317, -0.0133147, 0.0133147
6: 0.9943414, 1.0042117, 0.9943414, 1.0042117, -0.0098702, 0.0098702
7: 0.0073448, 0.0201244, 0.0073448, 0.0201244, -0.0093050, 0.0093066
8: 0.0018158, 0.0072932, 0.0018158, 0.0072932, -0.0054774, 0.0054774
9: -0.0276300, -0.0138944, -0.0276300, -0.0138944, -0.0137356, 0.0137356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 188

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0070847, upper bound: 0.0070914
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0070835, upper bound: 0.0070957
time: 1.23 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037834, 0.0086199, 0.0037834, 0.0086199, -0.0048365, 0.0048365
1: -0.0002308, 0.0047780, -0.0002308, 0.0047780, -0.0050088, 0.0050088
2: -0.0257978, -0.0053192, -0.0257978, -0.0053192, -0.0204786, 0.0204786
3: -0.0020375, 0.0080109, -0.0020375, 0.0080109, -0.0100484, 0.0100484
4: 0.0114507, 0.0185106, 0.0114507, 0.0185106, -0.0070599, 0.0070599
5: -0.0032830, 0.0100317, -0.0032830, 0.0100317, -0.0133147, 0.0133147
6: 0.9943414, 1.0042117, 0.9943414, 1.0042117, -0.0098702, 0.0098702
7: 0.0073448, 0.0201244, 0.0073448, 0.0201244, -0.0092908, 0.0093208
8: 0.0018158, 0.0072932, 0.0018158, 0.0072932, -0.0054774, 0.0054774
9: -0.0276300, -0.0138944, -0.0276300, -0.0138944, -0.0137356, 0.0137356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0064507, upper bound: 0.0064416
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0064507, upper bound: 0.0064416
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037834, 0.0086199, 0.0037834, 0.0086199, -0.0048365, 0.0048365
1: -0.0002308, 0.0047780, -0.0002308, 0.0047780, -0.0050088, 0.0050088
2: -0.0257978, -0.0053192, -0.0257978, -0.0053192, -0.0204786, 0.0204786
3: -0.0020375, 0.0080109, -0.0020375, 0.0080109, -0.0100484, 0.0100484
4: 0.0114507, 0.0185106, 0.0114507, 0.0185106, -0.0070599, 0.0070599
5: -0.0032830, 0.0100317, -0.0032830, 0.0100317, -0.0133147, 0.0133147
6: 0.9943414, 1.0042117, 0.9943414, 1.0042117, -0.0098702, 0.0098702
7: 0.0073448, 0.0201244, 0.0073448, 0.0201244, -0.0092769, 0.0093371
8: 0.0018158, 0.0072932, 0.0018158, 0.0072932, -0.0054774, 0.0054774
9: -0.0276300, -0.0138944, -0.0276300, -0.0138944, -0.0137356, 0.0137356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 165

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0070908, upper bound: 0.0071063
time: 1.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0070907, upper bound: 0.0071065
time: 1.12 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037834, 0.0086199, 0.0037834, 0.0086199, -0.0048365, 0.0048365
1: -0.0002308, 0.0047780, -0.0002308, 0.0047780, -0.0050088, 0.0050088
2: -0.0257978, -0.0053192, -0.0257978, -0.0053192, -0.0204786, 0.0204786
3: -0.0020375, 0.0080109, -0.0020375, 0.0080109, -0.0100484, 0.0100484
4: 0.0114507, 0.0185106, 0.0114507, 0.0185106, -0.0070599, 0.0070599
5: -0.0032830, 0.0100317, -0.0032830, 0.0100317, -0.0133147, 0.0133147
6: 0.9943414, 1.0042117, 0.9943414, 1.0042117, -0.0098702, 0.0098702
7: 0.0073448, 0.0201244, 0.0073448, 0.0201244, -0.0094193, 0.0094002
8: 0.0018158, 0.0072932, 0.0018158, 0.0072932, -0.0054774, 0.0054774
9: -0.0276300, -0.0138944, -0.0276300, -0.0138944, -0.0137356, 0.0137356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 188

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0071801, upper bound: 0.0071751
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0071764, upper bound: 0.0071787
time: 1.16 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037834, 0.0086199, 0.0037834, 0.0086199, -0.0048365, 0.0048365
1: -0.0002308, 0.0047780, -0.0002308, 0.0047780, -0.0050088, 0.0050088
2: -0.0257978, -0.0053192, -0.0257978, -0.0053192, -0.0204786, 0.0204786
3: -0.0020375, 0.0080109, -0.0020375, 0.0080109, -0.0100484, 0.0100484
4: 0.0114507, 0.0185106, 0.0114507, 0.0185106, -0.0070599, 0.0070599
5: -0.0032830, 0.0100317, -0.0032830, 0.0100317, -0.0133147, 0.0133147
6: 0.9943414, 1.0042117, 0.9943414, 1.0042117, -0.0098702, 0.0098702
7: 0.0073448, 0.0201244, 0.0073448, 0.0201244, -0.0094169, 0.0094033
8: 0.0018158, 0.0072932, 0.0018158, 0.0072932, -0.0054774, 0.0054774
9: -0.0276300, -0.0138944, -0.0276300, -0.0138944, -0.0137356, 0.0137356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 75

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0071051, upper bound: 0.0070920
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0070919, upper bound: 0.0071036
time: 1.13 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 3.60 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.60
Output dim: 6, lower bound: -0.0070196, upper bound: 0.0069875
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.60
Output dim: 6, lower bound: -0.0070196, upper bound: 0.0069875
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.60
Output dim: 6, lower bound: -0.0071248, upper bound: 0.0071017
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.60
Output dim: 6, lower bound: -0.0071248, upper bound: 0.0071017
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.60
Output dim: 6, lower bound: -0.0071322, upper bound: 0.0070970
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.60
Output dim: 6, lower bound: -0.0071322, upper bound: 0.0070968
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.60
Output dim: 6, lower bound: -0.0071756, upper bound: 0.0071500
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.60
Output dim: 6, lower bound: -0.0071748, upper bound: 0.0071503
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.60
Output dim: 6, lower bound: -0.0065247, upper bound: 0.0065069
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.60
Output dim: 6, lower bound: -0.0065247, upper bound: 0.0065069
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.60
Output dim: 6, lower bound: -0.0071429, upper bound: 0.0071120
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.60
Output dim: 6, lower bound: -0.0071425, upper bound: 0.0071122
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.60
Output dim: 6, lower bound: -0.0070978, upper bound: 0.0070820
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.60
Output dim: 6, lower bound: -0.0070940, upper bound: 0.0070830
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.60
Output dim: 6, lower bound: -0.0070847, upper bound: 0.0070914
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.60
Output dim: 6, lower bound: -0.0070835, upper bound: 0.0070957
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.60
Output dim: 6, lower bound: -0.0064507, upper bound: 0.0064416
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.60
Output dim: 6, lower bound: -0.0064507, upper bound: 0.0064416
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.60
Output dim: 6, lower bound: -0.0070908, upper bound: 0.0071063
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.60
Output dim: 6, lower bound: -0.0070907, upper bound: 0.0071065
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.60
Output dim: 6, lower bound: -0.0071801, upper bound: 0.0071751
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.60
Output dim: 6, lower bound: -0.0071764, upper bound: 0.0071787
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.60
Output dim: 6, lower bound: -0.0071051, upper bound: 0.0070920
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.60
Output dim: 6, lower bound: -0.0070919, upper bound: 0.0071036

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037834, 0.0086199, 0.0037834, 0.0086199, -0.0048365, 0.0048365
1: -0.0002308, 0.0047780, -0.0002308, 0.0047780, -0.0050088, 0.0050088
2: -0.0257978, -0.0053192, -0.0257978, -0.0053192, -0.0204786, 0.0204786
3: -0.0020375, 0.0080109, -0.0020375, 0.0080109, -0.0100484, 0.0100484
4: 0.0114507, 0.0185106, 0.0114507, 0.0185106, -0.0070599, 0.0070599
5: -0.0032830, 0.0100317, -0.0032830, 0.0100317, -0.0133147, 0.0133147
6: 0.9943414, 1.0042117, 0.9943414, 1.0042117, -0.0098702, 0.0098702
7: 0.0073448, 0.0201244, 0.0073448, 0.0201244, -0.0091026, 0.0090961
8: 0.0018158, 0.0072932, 0.0018158, 0.0072932, -0.0054774, 0.0054774
9: -0.0276300, -0.0138944, -0.0276300, -0.0138944, -0.0137356, 0.0137356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 153

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0068355, upper bound: 0.0068207
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0068355, upper bound: 0.0068207
time: 1.13 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037834, 0.0086199, 0.0037834, 0.0086199, -0.0048365, 0.0048365
1: -0.0002308, 0.0047780, -0.0002308, 0.0047780, -0.0050088, 0.0050088
2: -0.0257978, -0.0053192, -0.0257978, -0.0053192, -0.0204786, 0.0204786
3: -0.0020375, 0.0080109, -0.0020375, 0.0080109, -0.0100484, 0.0100484
4: 0.0114507, 0.0185106, 0.0114507, 0.0185106, -0.0070599, 0.0070599
5: -0.0032830, 0.0100317, -0.0032830, 0.0100317, -0.0133147, 0.0133147
6: 0.9943414, 1.0042117, 0.9943414, 1.0042117, -0.0098702, 0.0098702
7: 0.0073448, 0.0201244, 0.0073448, 0.0201244, -0.0090981, 0.0090971
8: 0.0018158, 0.0072932, 0.0018158, 0.0072932, -0.0054774, 0.0054774
9: -0.0276300, -0.0138944, -0.0276300, -0.0138944, -0.0137356, 0.0137356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 75

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0064922, upper bound: 0.0064848
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0064922, upper bound: 0.0064848
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037834, 0.0086199, 0.0037834, 0.0086199, -0.0048365, 0.0048365
1: -0.0002308, 0.0047780, -0.0002308, 0.0047780, -0.0050088, 0.0050088
2: -0.0257978, -0.0053192, -0.0257978, -0.0053192, -0.0204786, 0.0204786
3: -0.0020375, 0.0080109, -0.0020375, 0.0080109, -0.0100484, 0.0100484
4: 0.0114507, 0.0185106, 0.0114507, 0.0185106, -0.0070599, 0.0070599
5: -0.0032830, 0.0100317, -0.0032830, 0.0100317, -0.0133147, 0.0133147
6: 0.9943414, 1.0042117, 0.9943414, 1.0042117, -0.0098702, 0.0098702
7: 0.0073448, 0.0201244, 0.0073448, 0.0201244, -0.0091170, 0.0090832
8: 0.0018158, 0.0072932, 0.0018158, 0.0072932, -0.0054774, 0.0054774
9: -0.0276300, -0.0138944, -0.0276300, -0.0138944, -0.0137356, 0.0137356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0068414, upper bound: 0.0068168
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0068414, upper bound: 0.0068168
time: 1.19 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037834, 0.0086199, 0.0037834, 0.0086199, -0.0048365, 0.0048365
1: -0.0002308, 0.0047780, -0.0002308, 0.0047780, -0.0050088, 0.0050088
2: -0.0257978, -0.0053192, -0.0257978, -0.0053192, -0.0204786, 0.0204786
3: -0.0020375, 0.0080109, -0.0020375, 0.0080109, -0.0100484, 0.0100484
4: 0.0114507, 0.0185106, 0.0114507, 0.0185106, -0.0070599, 0.0070599
5: -0.0032830, 0.0100317, -0.0032830, 0.0100317, -0.0133147, 0.0133147
6: 0.9943414, 1.0042117, 0.9943414, 1.0042117, -0.0098702, 0.0098702
7: 0.0073448, 0.0201244, 0.0073448, 0.0201244, -0.0091144, 0.0090844
8: 0.0018158, 0.0072932, 0.0018158, 0.0072932, -0.0054774, 0.0054774
9: -0.0276300, -0.0138944, -0.0276300, -0.0138944, -0.0137356, 0.0137356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 75

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0064964, upper bound: 0.0064852
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0064964, upper bound: 0.0064852
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037834, 0.0086199, 0.0037834, 0.0086199, -0.0048365, 0.0048365
1: -0.0002308, 0.0047780, -0.0002308, 0.0047780, -0.0050088, 0.0050088
2: -0.0257978, -0.0053192, -0.0257978, -0.0053192, -0.0204786, 0.0204786
3: -0.0020375, 0.0080109, -0.0020375, 0.0080109, -0.0100484, 0.0100484
4: 0.0114507, 0.0185106, 0.0114507, 0.0185106, -0.0070599, 0.0070599
5: -0.0032830, 0.0100317, -0.0032830, 0.0100317, -0.0133147, 0.0133147
6: 0.9943414, 1.0042117, 0.9943414, 1.0042117, -0.0098702, 0.0098702
7: 0.0073448, 0.0201244, 0.0073448, 0.0201244, -0.0091958, 0.0091903
8: 0.0018158, 0.0072932, 0.0018158, 0.0072932, -0.0054774, 0.0054774
9: -0.0276300, -0.0138944, -0.0276300, -0.0138944, -0.0137356, 0.0137356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0059500, upper bound: 0.0059435
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0059500, upper bound: 0.0059435
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037834, 0.0086199, 0.0037834, 0.0086199, -0.0048365, 0.0048365
1: -0.0002308, 0.0047780, -0.0002308, 0.0047780, -0.0050088, 0.0050088
2: -0.0257978, -0.0053192, -0.0257978, -0.0053192, -0.0204786, 0.0204786
3: -0.0020375, 0.0080109, -0.0020375, 0.0080109, -0.0100484, 0.0100484
4: 0.0114507, 0.0185106, 0.0114507, 0.0185106, -0.0070599, 0.0070599
5: -0.0032830, 0.0100317, -0.0032830, 0.0100317, -0.0133147, 0.0133147
6: 0.9943414, 1.0042117, 0.9943414, 1.0042117, -0.0098702, 0.0098702
7: 0.0073448, 0.0201244, 0.0073448, 0.0201244, -0.0091932, 0.0091935
8: 0.0018158, 0.0072932, 0.0018158, 0.0072932, -0.0054774, 0.0054774
9: -0.0276300, -0.0138944, -0.0276300, -0.0138944, -0.0137356, 0.0137356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 75

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0071138, upper bound: 0.0070934
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0071138, upper bound: 0.0070934
time: 1.21 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037834, 0.0086199, 0.0037834, 0.0086199, -0.0048365, 0.0048365
1: -0.0002308, 0.0047780, -0.0002308, 0.0047780, -0.0050088, 0.0050088
2: -0.0257978, -0.0053192, -0.0257978, -0.0053192, -0.0204786, 0.0204786
3: -0.0020375, 0.0080109, -0.0020375, 0.0080109, -0.0100484, 0.0100484
4: 0.0114507, 0.0185106, 0.0114507, 0.0185106, -0.0070599, 0.0070599
5: -0.0032830, 0.0100317, -0.0032830, 0.0100317, -0.0133147, 0.0133147
6: 0.9943414, 1.0042117, 0.9943414, 1.0042117, -0.0098702, 0.0098702
7: 0.0073448, 0.0201244, 0.0073448, 0.0201244, -0.0091541, 0.0091230
8: 0.0018158, 0.0072932, 0.0018158, 0.0072932, -0.0054774, 0.0054774
9: -0.0276300, -0.0138944, -0.0276300, -0.0138944, -0.0137356, 0.0137356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 75

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0068600, upper bound: 0.0068408
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0068600, upper bound: 0.0068408
time: 1.13 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037834, 0.0086199, 0.0037834, 0.0086199, -0.0048365, 0.0048365
1: -0.0002308, 0.0047780, -0.0002308, 0.0047780, -0.0050088, 0.0050088
2: -0.0257978, -0.0053192, -0.0257978, -0.0053192, -0.0204786, 0.0204786
3: -0.0020375, 0.0080109, -0.0020375, 0.0080109, -0.0100484, 0.0100484
4: 0.0114507, 0.0185106, 0.0114507, 0.0185106, -0.0070599, 0.0070599
5: -0.0032830, 0.0100317, -0.0032830, 0.0100317, -0.0133147, 0.0133147
6: 0.9943414, 1.0042117, 0.9943414, 1.0042117, -0.0098702, 0.0098702
7: 0.0073448, 0.0201244, 0.0073448, 0.0201244, -0.0091392, 0.0091370
8: 0.0018158, 0.0072932, 0.0018158, 0.0072932, -0.0054774, 0.0054774
9: -0.0276300, -0.0138944, -0.0276300, -0.0138944, -0.0137356, 0.0137356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0068592, upper bound: 0.0068415
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0068592, upper bound: 0.0068415
time: 1.27 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037834, 0.0086199, 0.0037834, 0.0086199, -0.0048365, 0.0048365
1: -0.0002308, 0.0047780, -0.0002308, 0.0047780, -0.0050088, 0.0050088
2: -0.0257978, -0.0053192, -0.0257978, -0.0053192, -0.0204786, 0.0204786
3: -0.0020375, 0.0080109, -0.0020375, 0.0080109, -0.0100484, 0.0100484
4: 0.0114507, 0.0185106, 0.0114507, 0.0185106, -0.0070599, 0.0070599
5: -0.0032830, 0.0100317, -0.0032830, 0.0100317, -0.0133147, 0.0133147
6: 0.9943414, 1.0042117, 0.9943414, 1.0042117, -0.0098702, 0.0098702
7: 0.0073448, 0.0201244, 0.0073448, 0.0201244, -0.0092436, 0.0092997
8: 0.0018158, 0.0072932, 0.0018158, 0.0072932, -0.0054774, 0.0054774
9: -0.0276300, -0.0138944, -0.0276300, -0.0138944, -0.0137356, 0.0137356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0070505, upper bound: 0.0070261
time: 1.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0070189, upper bound: 0.0070660
time: 1.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037834, 0.0086199, 0.0037834, 0.0086199, -0.0048365, 0.0048365
1: -0.0002308, 0.0047780, -0.0002308, 0.0047780, -0.0050088, 0.0050088
2: -0.0257978, -0.0053192, -0.0257978, -0.0053192, -0.0204786, 0.0204786
3: -0.0020375, 0.0080109, -0.0020375, 0.0080109, -0.0100484, 0.0100484
4: 0.0114507, 0.0185106, 0.0114507, 0.0185106, -0.0070599, 0.0070599
5: -0.0032830, 0.0100317, -0.0032830, 0.0100317, -0.0133147, 0.0133147
6: 0.9943414, 1.0042117, 0.9943414, 1.0042117, -0.0098702, 0.0098702
7: 0.0073448, 0.0201244, 0.0073448, 0.0201244, -0.0092411, 0.0093039
8: 0.0018158, 0.0072932, 0.0018158, 0.0072932, -0.0054774, 0.0054774
9: -0.0276300, -0.0138944, -0.0276300, -0.0138944, -0.0137356, 0.0137356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0064284, upper bound: 0.0064437
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0064284, upper bound: 0.0064437
time: 0.94 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037834, 0.0086199, 0.0037834, 0.0086199, -0.0048365, 0.0048365
1: -0.0002308, 0.0047780, -0.0002308, 0.0047780, -0.0050088, 0.0050088
2: -0.0257978, -0.0053192, -0.0257978, -0.0053192, -0.0204786, 0.0204786
3: -0.0020375, 0.0080109, -0.0020375, 0.0080109, -0.0100484, 0.0100484
4: 0.0114507, 0.0185106, 0.0114507, 0.0185106, -0.0070599, 0.0070599
5: -0.0032830, 0.0100317, -0.0032830, 0.0100317, -0.0133147, 0.0133147
6: 0.9943414, 1.0042117, 0.9943414, 1.0042117, -0.0098702, 0.0098702
7: 0.0073448, 0.0201244, 0.0073448, 0.0201244, -0.0094247, 0.0094043
8: 0.0018158, 0.0072932, 0.0018158, 0.0072932, -0.0054774, 0.0054774
9: -0.0276300, -0.0138944, -0.0276300, -0.0138944, -0.0137356, 0.0137356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 75

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0065610, upper bound: 0.0065572
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0065610, upper bound: 0.0065572
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037834, 0.0086199, 0.0037834, 0.0086199, -0.0048365, 0.0048365
1: -0.0002308, 0.0047780, -0.0002308, 0.0047780, -0.0050088, 0.0050088
2: -0.0257978, -0.0053192, -0.0257978, -0.0053192, -0.0204786, 0.0204786
3: -0.0020375, 0.0080109, -0.0020375, 0.0080109, -0.0100484, 0.0100484
4: 0.0114507, 0.0185106, 0.0114507, 0.0185106, -0.0070599, 0.0070599
5: -0.0032830, 0.0100317, -0.0032830, 0.0100317, -0.0133147, 0.0133147
6: 0.9943414, 1.0042117, 0.9943414, 1.0042117, -0.0098702, 0.0098702
7: 0.0073448, 0.0201244, 0.0073448, 0.0201244, -0.0094232, 0.0094057
8: 0.0018158, 0.0072932, 0.0018158, 0.0072932, -0.0054774, 0.0054774
9: -0.0276300, -0.0138944, -0.0276300, -0.0138944, -0.0137356, 0.0137356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0070847, upper bound: 0.0070753
time: 1.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0070739, upper bound: 0.0070869
time: 1.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037834, 0.0086199, 0.0037834, 0.0086199, -0.0048365, 0.0048365
1: -0.0002308, 0.0047780, -0.0002308, 0.0047780, -0.0050088, 0.0050088
2: -0.0257978, -0.0053192, -0.0257978, -0.0053192, -0.0204786, 0.0204786
3: -0.0020375, 0.0080109, -0.0020375, 0.0080109, -0.0100484, 0.0100484
4: 0.0114507, 0.0185106, 0.0114507, 0.0185106, -0.0070599, 0.0070599
5: -0.0032830, 0.0100317, -0.0032830, 0.0100317, -0.0133147, 0.0133147
6: 0.9943414, 1.0042117, 0.9943414, 1.0042117, -0.0098702, 0.0098702
7: 0.0073448, 0.0201244, 0.0073448, 0.0201244, -0.0092888, 0.0092595
8: 0.0018158, 0.0072932, 0.0018158, 0.0072932, -0.0054774, 0.0054774
9: -0.0276300, -0.0138944, -0.0276300, -0.0138944, -0.0137356, 0.0137356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 188

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0070885, upper bound: 0.0070746
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0070848, upper bound: 0.0070753
time: 1.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037834, 0.0086199, 0.0037834, 0.0086199, -0.0048365, 0.0048365
1: -0.0002308, 0.0047780, -0.0002308, 0.0047780, -0.0050088, 0.0050088
2: -0.0257978, -0.0053192, -0.0257978, -0.0053192, -0.0204786, 0.0204786
3: -0.0020375, 0.0080109, -0.0020375, 0.0080109, -0.0100484, 0.0100484
4: 0.0114507, 0.0185106, 0.0114507, 0.0185106, -0.0070599, 0.0070599
5: -0.0032830, 0.0100317, -0.0032830, 0.0100317, -0.0133147, 0.0133147
6: 0.9943414, 1.0042117, 0.9943414, 1.0042117, -0.0098702, 0.0098702
7: 0.0073448, 0.0201244, 0.0073448, 0.0201244, -0.0092734, 0.0092734
8: 0.0018158, 0.0072932, 0.0018158, 0.0072932, -0.0054774, 0.0054774
9: -0.0276300, -0.0138944, -0.0276300, -0.0138944, -0.0137356, 0.0137356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 153

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 75

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0064895, upper bound: 0.0064914
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0064895, upper bound: 0.0064914
time: 1.12 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 3.68 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.68
Output dim: 6, lower bound: -0.0068355, upper bound: 0.0068207
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.68
Output dim: 6, lower bound: -0.0068355, upper bound: 0.0068207
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.68
Output dim: 6, lower bound: -0.0064922, upper bound: 0.0064848
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.68
Output dim: 6, lower bound: -0.0064922, upper bound: 0.0064848
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.68
Output dim: 6, lower bound: -0.0068414, upper bound: 0.0068168
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.68
Output dim: 6, lower bound: -0.0068414, upper bound: 0.0068168
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.68
Output dim: 6, lower bound: -0.0064964, upper bound: 0.0064852
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.68
Output dim: 6, lower bound: -0.0064964, upper bound: 0.0064852
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.68
Output dim: 6, lower bound: -0.0059500, upper bound: 0.0059435
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.68
Output dim: 6, lower bound: -0.0059500, upper bound: 0.0059435
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.68
Output dim: 6, lower bound: -0.0071138, upper bound: 0.0070934
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.68
Output dim: 6, lower bound: -0.0071138, upper bound: 0.0070934
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.68
Output dim: 6, lower bound: -0.0068600, upper bound: 0.0068408
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.68
Output dim: 6, lower bound: -0.0068600, upper bound: 0.0068408
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.68
Output dim: 6, lower bound: -0.0068592, upper bound: 0.0068415
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.68
Output dim: 6, lower bound: -0.0068592, upper bound: 0.0068415
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.68
Output dim: 6, lower bound: -0.0070505, upper bound: 0.0070261
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.68
Output dim: 6, lower bound: -0.0070189, upper bound: 0.0070660
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.68
Output dim: 6, lower bound: -0.0064284, upper bound: 0.0064437
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.68
Output dim: 6, lower bound: -0.0064284, upper bound: 0.0064437
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.68
Output dim: 6, lower bound: -0.0065610, upper bound: 0.0065572
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.68
Output dim: 6, lower bound: -0.0065610, upper bound: 0.0065572
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.68
Output dim: 6, lower bound: -0.0070847, upper bound: 0.0070753
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.68
Output dim: 6, lower bound: -0.0070739, upper bound: 0.0070869
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.68
Output dim: 6, lower bound: -0.0070885, upper bound: 0.0070746
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.68
Output dim: 6, lower bound: -0.0070848, upper bound: 0.0070753
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.68
Output dim: 6, lower bound: -0.0064895, upper bound: 0.0064914
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.68
Output dim: 6, lower bound: -0.0064895, upper bound: 0.0064914

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0037834, 0.0086199, 0.0037834, 0.0086199, -0.0048365, 0.0048365
1: -0.0002308, 0.0047780, -0.0002308, 0.0047780, -0.0050088, 0.0050088
2: -0.0257978, -0.0053192, -0.0257978, -0.0053192, -0.0204786, 0.0204786
3: -0.0020375, 0.0080109, -0.0020375, 0.0080109, -0.0100484, 0.0100484
4: 0.0114507, 0.0185106, 0.0114507, 0.0185106, -0.0070599, 0.0070599
5: -0.0032830, 0.0100317, -0.0032830, 0.0100317, -0.0133147, 0.0133147
6: 0.9943414, 1.0042117, 0.9943414, 1.0042117, -0.0098702, 0.0098702
7: 0.0073448, 0.0201244, 0.0073448, 0.0201244, -0.0090644, 0.0090638
8: 0.0018158, 0.0072932, 0.0018158, 0.0072932, -0.0054774, 0.0054774
9: -0.0276300, -0.0138944, -0.0276300, -0.0138944, -0.0137356, 0.0137356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=247
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 75

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0064830, upper bound: 0.0064762
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0064830, upper bound: 0.0064762
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0037834, 0.0086199, 0.0037834, 0.0086199, -0.0048365, 0.0048365
1: -0.0002308, 0.0047780, -0.0002308, 0.0047780, -0.0050088, 0.0050088
2: -0.0257978, -0.0053192, -0.0257978, -0.0053192, -0.0204786, 0.0204786
3: -0.0020375, 0.0080109, -0.0020375, 0.0080109, -0.0100484, 0.0100484
4: 0.0114507, 0.0185106, 0.0114507, 0.0185106, -0.0070599, 0.0070599
5: -0.0032830, 0.0100317, -0.0032830, 0.0100317, -0.0133147, 0.0133147
6: 0.9943414, 1.0042117, 0.9943414, 1.0042117, -0.0098702, 0.0098702
7: 0.0073448, 0.0201244, 0.0073448, 0.0201244, -0.0090605, 0.0090648
8: 0.0018158, 0.0072932, 0.0018158, 0.0072932, -0.0054774, 0.0054774
9: -0.0276300, -0.0138944, -0.0276300, -0.0138944, -0.0137356, 0.0137356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=247
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=33, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 75

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0064830, upper bound: 0.0064762
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0064830, upper bound: 0.0064762
time: 1.05 seconds

## Summary of splitting (split count: 9)
- Time for RS candidates: 3.59 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 10, time: 3.59
Output dim: 6, lower bound: -0.0064830, upper bound: 0.0064762
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 10, time: 3.59
Output dim: 6, lower bound: -0.0064830, upper bound: 0.0064762
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 10, time: 3.59
Output dim: 6, lower bound: -0.0064830, upper bound: 0.0064762
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 10, time: 3.59
Output dim: 6, lower bound: -0.0064830, upper bound: 0.0064762

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 3.89 + 357.09 = 360.98 seconds
